import copy
import os
import time

import cv2
import numpy as np
import torch
import torchvision.ops.boxes as bops
from torch import nn
from torchvision.utils import save_image

from S2DNet.s2dnet import S2DNet
from helpers.torch_helpers import write_renders
from main_settings import g_ext_folder
from models.encoder import Encoder, qmult, qnorm
from models.initial_mesh import generate_initial_mesh, generate_face_features
from models.kaolin_wrapper import load_obj, write_obj_mesh
from models.loss import FMOLoss
from models.rendering import RenderingKaolin
from segmentations import PrecomputedTracker, get_bbox, OSTracker, MyTracker, CSRTrack, create_mask_from_string
from utils import write_video, segment2bbox


class Tracking6D():
    def __init__(self, config, device, write_folder, file0, bbox0, init_mask=None):
        self.write_folder = write_folder
        self.config = config.copy()
        self.config["fmo_steps"] = 1
        self.device = device
        torch.backends.cudnn.benchmark = True
        if type(bbox0) is dict:
            self.tracker = PrecomputedTracker(self.config["image_downsample"],self.config["max_width"],bbox0,self.config["grabcut"])
        else:
            if self.config["tracker_type"] is 'csrt':
                self.tracker = CSRTrack(self.config["image_downsample"],self.config["max_width"],self.config["grabcut"])
            elif self.config["tracker_type"] is 'ostrack':
                self.tracker = OSTracker(self.config["image_downsample"],self.config["max_width"],self.config["grabcut"])
            else: # d3s
                self.tracker = MyTracker(self.config["image_downsample"],self.config["max_width"],self.config["grabcut"])
        if self.config["features"] == 'deep':
            self.net = S2DNet(device=device,checkpoint_path=g_ext_folder).to(device)
            self.feat = lambda x: self.net(x[0])[0][None][:,:,:64]
        else:
            self.feat = lambda x: x
        self.images, self.segments, self.config["image_downsample"] = self.tracker.init_bbox(file0, bbox0, init_mask)
        self.images, self.segments = self.images[None].to(self.device), self.segments[None].to(self.device)
        self.images_feat = self.feat(self.images).detach()
        shape = self.segments.shape
        prot = self.config["shapes"][0]
        if not config["init_shape"] is False:
            mesh = load_obj(config["init_shape"])
            ivertices = mesh.vertices.numpy()
            ivertices = ivertices - ivertices.mean(0)
            ivertices = ivertices / ivertices.max()
            faces = mesh.faces.numpy().copy()
            iface_features = generate_face_features(ivertices, faces)
        elif prot == 'sphere':
            ivertices, faces, iface_features = generate_initial_mesh(self.config["mesh_size"])
        else:
            mesh = load_obj(os.path.join('/cluster/home/denysr/src/ShapeFromBlur/prototypes',prot+'.obj'))
            ivertices = mesh.vertices.numpy()
            faces = mesh.faces.numpy().copy()
            iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()
        self.faces = faces
        self.rendering = RenderingKaolin(self.config, self.faces, shape[-1], shape[-2]).to(self.device)
        self.encoder = Encoder(self.config, ivertices, faces, iface_features, shape[-1], shape[-2], self.images_feat.shape[2]).to(self.device)
        all_parameters = list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(all_parameters, lr = self.config["learning_rate"])
        self.encoder.train()
        self.loss_function = FMOLoss(self.config, ivertices, faces).to(self.device)
        if self.config["features"] == 'deep':
            config = self.config.copy()
            config["features"] = 'rgb'
            self.rgb_encoder = Encoder(config, ivertices, faces, iface_features, shape[-1], shape[-2], 3).to(self.device)
            rgb_parameters = list(self.rgb_encoder.parameters())[-1:]
            self.rgb_optimizer = torch.optim.Adam(rgb_parameters, lr = self.config["learning_rate"])
            self.rgb_encoder.train()
            config["loss_laplacian_weight"] = 0
            config["loss_tv_weight"] = 1.0
            config["loss_iou_weight"] = 0
            config["loss_dist_weight"] = 0
            config["loss_qt_weight"] = 0
            self.rgb_loss_function = FMOLoss(config, ivertices, faces).to(self.device)
        if self.config["verbose"]:
            print('Total params {}'.format(sum(p.numel() for p in self.encoder.parameters())))
        self.best_model = {}
        self.best_model["value"] = 100
        self.best_model["face_features"] = self.encoder.face_features.detach().clone()
        self.best_model["faces"] = faces
        self.best_model["encoder"] = None
        self.keyframes = [0]


    def run_tracking(self, files, bboxes, depths_dict=None):
        all_input = cv2.VideoWriter(os.path.join(self.write_folder,'all_input.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 10, (self.images.shape[4], self.images.shape[3]),True)
        all_segm = cv2.VideoWriter(os.path.join(self.write_folder,'all_segm.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 10, (self.images.shape[4], self.images.shape[3]),True)
        all_proj = cv2.VideoWriter(os.path.join(self.write_folder,'all_proj.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 10, (self.images.shape[4], self.images.shape[3]),True)
        all_proj_filtered = cv2.VideoWriter(os.path.join(self.write_folder,'all_proj_filtered.avi'),cv2.VideoWriter_fourcc(*"MJPG"), 10, (self.images.shape[4], self.images.shape[3]),True)
        baseline_iou = -np.ones((files.shape[0]-1,1))
        our_iou = -np.ones((files.shape[0]-1,1))
        our_losses = -np.ones((files.shape[0]-1,1))
        self.config["loss_rgb_weight"] = 0
        removed_count = 0
        silh_losses = [0]
        b0 = None
        for stepi in range(1,self.config["input_frames"]):
            image_raw, segment = self.tracker.next(files[stepi])
            image, segment = image_raw[None].to(self.device), segment[None].to(self.device)
            if b0 is not None:
                segment_clean = segment*0
                segment_clean[:,:,:,b0[0]:b0[1],b0[2]:b0[3]] = segment[:,:,:,b0[0]:b0[1],b0[2]:b0[3]]
                segment_clean[:,:,0] = segment[:,:,0]
                segment = segment_clean
            self.images = torch.cat( (self.images, image), 1)     
            image_feat = self.feat(image).detach()
            self.images_feat = torch.cat( (self.images_feat, image_feat), 1)        
            self.segments = torch.cat( (self.segments, segment), 1)        
            self.keyframes.append(stepi)
            start = time.time()
            b0 = get_bbox(self.segments)
            self.rendering = RenderingKaolin(self.config, self.faces, b0[3]-b0[2], b0[1]-b0[0]).to(self.device)

            self.encoder.offsets[:,:,stepi,:3] = (self.encoder.used_tran[:,:,stepi-1]+self.encoder.offsets[:,:,stepi-1,:3])
            self.encoder.offsets[:,0,stepi,3:] = qmult(qnorm(self.encoder.used_quat[:,stepi-1]),qnorm(self.encoder.offsets[:,0,stepi-1,3:]))
            encoder_backup = copy.deepcopy(self.encoder.state_dict())

            self.apply(self.images_feat[:,:,:,b0[0]:b0[1],b0[2]:b0[3]], self.segments[:,:,:,b0[0]:b0[1],b0[2]:b0[3]], self.keyframes, b0)
            silh_losses = np.array(self.best_model["losses"]["silh"])

            our_losses[stepi-1] = silh_losses[-1]
            print('Elapsed time in seconds: ', time.time() - start)
            if silh_losses[-1] < 0.8:
                self.encoder.used_tran[:,:,stepi] = self.encoder.translation[:,:,stepi].detach()
                self.encoder.used_quat[:,stepi] = self.encoder.quaternion[:,stepi].detach()
           
            if self.config["write_results"]:
                if self.config["features"] == 'deep':
                    self.rgb_apply(self.images[:,:,:,b0[0]:b0[1],b0[2]:b0[3]], self.segments[:,:,:,b0[0]:b0[1],b0[2]:b0[3]], self.keyframes, b0)
                    tex = nn.Sigmoid()(self.rgb_encoder.texture_map)
                with torch.no_grad():
                    translation, quaternion, vertices, texture_maps, lights, tdiff, qdiff = self.encoder(self.keyframes)
                    if self.config["features"] == 'rgb':
                        tex = texture_maps
                    feat_renders_crop = self.rendering(translation, quaternion, vertices, self.encoder.face_features, texture_maps, lights, True)
                    depth_map_crop = feat_renders_crop[:,:,:,-1]
                    feat_renders_crop = feat_renders_crop[:,:,:,:-1]
                    feat_renders_crop = torch.cat((feat_renders_crop[:,:,:,:3],feat_renders_crop[:,:,:,-1:]),3)
                    renders_crop = self.rendering(translation, quaternion, vertices, self.encoder.face_features, tex, lights)
                    renders_crop = torch.cat((renders_crop[:,:,:,:3],renders_crop[:,:,:,-1:]),3)
                    renders = torch.zeros(renders_crop.shape[:4]+self.images_feat.shape[-2:]).to(self.device)
                    renders[:,:,:,:,b0[0]:b0[1],b0[2]:b0[3]] = renders_crop
                    write_renders(feat_renders_crop, self.write_folder, self.config["max_keyframes"]+1, ids=0)
                    write_renders(renders_crop, self.write_folder, self.config["max_keyframes"]+1, ids=1)
                    write_renders(torch.cat((self.images[:,:,None,:,b0[0]:b0[1],b0[2]:b0[3]],feat_renders_crop[:,:,:,-1:]),3), self.write_folder, self.config["max_keyframes"]+1, ids=2)
                    write_obj_mesh(vertices[0].cpu().numpy(), self.best_model["faces"], self.encoder.face_features[0].cpu().numpy(), os.path.join(self.write_folder,'mesh.obj'))
                    save_image(texture_maps[:,:3], os.path.join(self.write_folder,'tex_deep.png'))
                    save_image(tex, os.path.join(self.write_folder,'tex.png'))
                    write_video(renders[0,:,0,:3].detach().cpu().numpy().transpose(2,3,1,0), os.path.join(self.write_folder,'im_recon.avi'), fps=6)
                    write_video(self.images[0,:,:3].cpu().numpy().transpose(2,3,1,0), os.path.join(self.write_folder,'input.avi'), fps=6)
                    write_video((self.images[0,:,:3]*self.segments[0,:,1:2]).cpu().numpy().transpose(2,3,1,0), os.path.join(self.write_folder,'segments.avi'), fps=6)
                    for tmpi in range(renders.shape[1]):
                        img = self.images[0,tmpi,:3,b0[0]:b0[1],b0[2]:b0[3]]
                        seg = self.segments[0,:,1:2][tmpi,:,b0[0]:b0[1],b0[2]:b0[3]].clone()
                        save_image(seg, os.path.join(self.write_folder, 'imgs', 's{}.png'.format(tmpi)))
                        seg[seg == 0] = 0.35
                        save_image(img, os.path.join(self.write_folder, 'imgs', 'i{}.png'.format(tmpi)))
                        save_image(self.images_feat[0,tmpi,:3,b0[0]:b0[1],b0[2]:b0[3]], os.path.join(self.write_folder, 'imgs', 'if{}.png'.format(tmpi)))
                        save_image(torch.cat((img,seg),0), os.path.join(self.write_folder, 'imgs', 'is{}.png'.format(tmpi)))
                        save_image(renders_crop[0,tmpi,0,[3,3,3]], os.path.join(self.write_folder, 'imgs', 'm{}.png'.format(tmpi)))
                        save_image(renders_crop[0,tmpi,0,:], os.path.join(self.write_folder, 'imgs', 'r{}.png'.format(tmpi)))
                        save_image(feat_renders_crop[0,tmpi,0,:], os.path.join(self.write_folder, 'imgs', 'f{}.png'.format(tmpi)))
                    if type(bboxes) is dict or (bboxes[stepi][0] is 'm'):
                        gt_segm = None
                        if (not type(bboxes) is dict) and bboxes[stepi][0] is 'm':
                            m_, offset_ = create_mask_from_string(bboxes[stepi][1:].split(','))
                            gt_segm = segment[0,0,-1]*0
                            gt_segm[offset_[1]:offset_[1]+m_.shape[0], offset_[0]:offset_[0]+m_.shape[1]] = torch.from_numpy(m_)
                        elif stepi in bboxes:
                            gt_segm = self.tracker.process_segm(bboxes[stepi])[0].to(self.device)
                        if not gt_segm is None:
                            baseline_iou[stepi-1] = float((segment[0,0,-1]*gt_segm > 0).sum())/float(((segment[0,0,-1]+gt_segm) > 0).sum()+0.00001)
                            our_iou[stepi-1] = float((renders[0,-1,0,3]*gt_segm > 0).sum())/float(((renders[0,-1,0,3]+gt_segm) > 0).sum()+0.00001)
                    elif bboxes is not None:   
                        bbox = self.config["image_downsample"]*torch.tensor([bboxes[stepi]+[0,0,bboxes[stepi][0],bboxes[stepi][1]]])
                        baseline_iou[stepi-1] = bops.box_iou(bbox, torch.tensor([segment2bbox(segment[0,0,-1])], dtype=torch.float64))
                        our_iou[stepi-1] = bops.box_iou(bbox, torch.tensor([segment2bbox(renders[0,-1,0,3])], dtype=torch.float64))
                    print('Baseline IoU {}, our IoU {}'.format(baseline_iou[stepi-1], our_iou[stepi-1]))
                    np.savetxt(os.path.join(self.write_folder,'baseline_iou.txt'), baseline_iou, fmt='%.10f', delimiter='\n')
                    np.savetxt(os.path.join(self.write_folder,'iou.txt'), our_iou, fmt='%.10f', delimiter='\n')
                    np.savetxt(os.path.join(self.write_folder,'losses.txt'), our_losses, fmt='%.10f', delimiter='\n')
                    all_input.write((self.images[0,:,:3].clamp(min=0,max=1).cpu().numpy().transpose(2,3,1,0)[:,:,[2,1,0],-1] * 255).astype(np.uint8))
                    all_segm.write(((self.images[0,:,:3]*self.segments[0,:,1:2]).clamp(min=0,max=1).cpu().numpy().transpose(2,3,1,0)[:,:,[2,1,0],-1] * 255).astype(np.uint8))
                    all_proj.write((renders[0,:,0,:3].detach().clamp(min=0,max=1).cpu().numpy().transpose(2,3,1,0)[:,:,[2,1,0],-1] * 255).astype(np.uint8))
                    if silh_losses[-1] > 0.3:
                        renders[0,-1,0,3] = segment[0,0,-1]
                        renders[0,-1,0,:3] = self.images[0,-1,:3]*segment[0,0,-1]
                    all_proj_filtered.write((renders[0,:,0,:3].detach().clamp(min=0,max=1).cpu().numpy().transpose(2,3,1,0)[:,:,[2,1,0],-1] * 255).astype(np.uint8))
            keep_keyframes = (silh_losses <= 0.8)
            if silh_losses[-1] > 0.3:
                keep_keyframes[-1] = False
            keep_keyframes[np.argmin(silh_losses)] = True
            self.keyframes = (np.array(self.keyframes)[keep_keyframes]).tolist()
            self.images = self.images[:,keep_keyframes]
            self.images_feat = self.images_feat[:,keep_keyframes]
            self.segments = self.segments[:,keep_keyframes]
            if len(self.keyframes) >= 3:
                l1, _, _ = self.loss_function(renders[:,-1:], renders[:,-2:-1,0][:,:,[-1,-1]], renders[:,-2:-1,0,:3], vertices, texture_maps, tdiff, qdiff)
                l2, _, _ = self.loss_function(renders[:,-2:-1], renders[:,-3:-2,0][:,:,[-1,-1]], renders[:,-3:-2,0,:3], vertices, texture_maps, tdiff, qdiff)
                if l1["silh"][-1] < 0.7 and l2["silh"][-1] < 0.7 and removed_count < 30:
                    removed_count += 1
                    self.keyframes = self.keyframes[:-2] + [stepi]
                    self.images = torch.cat( (self.images[:,:-2], image), 1)
                    self.images_feat = torch.cat( (self.images_feat[:,:-2], image_feat), 1)
                    self.segments = torch.cat( (self.segments[:,:-2], segment), 1)
                else:
                    removed_count = 0
            if len(self.keyframes) > self.config["max_keyframes"]:
                self.keyframes = self.keyframes[-self.config["max_keyframes"]:]
                self.images = self.images[:,-self.config["max_keyframes"]:]
                self.images_feat = self.images_feat[:,-self.config["max_keyframes"]:]
                self.segments = self.segments[:,-self.config["max_keyframes"]:]
        all_input.release()
        all_segm.release()
        all_proj.release()
        all_proj_filtered.release()

        return self.best_model

    def apply(self, input_batch, segments, opt_frames = None, bounds = None):
        if self.config["write_results"]:
            save_image(input_batch[0,:,:3],os.path.join(self.write_folder,'im.png'), nrow=self.config["max_keyframes"]+1)
            save_image(torch.cat((input_batch[0,:,:3],segments[0,:,[1]]),1),os.path.join(self.write_folder,'segments.png'), nrow=self.config["max_keyframes"]+1)
            if self.config["weight_by_gradient"]:
                save_image(torch.cat((segments[0,:,[0,0,0]],0*input_batch[0,:,:1]+1),1),os.path.join(self.write_folder,'weights.png'))
        
        self.best_model["value"] = 100
        self.best_model["losses"] = None
        iters_without_change = 0
        for epoch in range(self.config["iterations"]):
            translation, quaternion, vertices, texture_maps, lights, tdiff, qdiff = self.encoder(opt_frames)
            renders = self.rendering(translation, quaternion, vertices, self.encoder.face_features, texture_maps, lights)
            losses_all, losses, jloss = self.loss_function(renders, segments, input_batch, vertices, texture_maps, tdiff, qdiff)

            if "model" in losses:
                model_loss = losses["model"].mean().item()
            else:
                model_loss = losses["silh"].mean().item()
            if self.config["verbose"] and epoch % 20 == 0:
                print("Epoch {:4d}".format(epoch+1), end =" ")
                for ls in losses:
                    print(", {} {:.3f}".format(ls,losses[ls].mean().item()), end =" ")
                print("; joint {:.3f}".format(jloss.item()))

            if model_loss < self.best_model["value"]:
                iters_without_change = 0
                self.best_model["value"] = model_loss
                self.best_model["losses"] = losses_all
                self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())
                if self.config["write_intermediate"]:
                    write_renders(torch.cat((renders[:,:,:,:3],renders[:,:,:,-1:]),3), self.write_folder, self.config["max_keyframes"]+1)
            else:
                iters_without_change += 1

            if self.config["loss_rgb_weight"] == 0:
                if epoch > 100 or model_loss < 0.1:
                    self.config["loss_rgb_weight"] = 1.0
                    self.best_model["value"] = 100
            else:
                if epoch > 50 and self.best_model["value"] < self.config["stop_value"] and iters_without_change > 10:
                    break
            if epoch < self.config["iterations"] - 1:
                jloss = jloss.mean()
                self.optimizer.zero_grad()
                jloss.backward()
                self.optimizer.step()
        self.encoder.load_state_dict(self.best_model["encoder"])

    def rgb_apply(self, input_batch, segments, opt_frames = None, bounds = None):
        self.best_model["value"] = 100
        model_state = self.rgb_encoder.state_dict()
        pretrained_dict = self.best_model["encoder"] 
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != "texture_map"}
        model_state.update(pretrained_dict)
        self.rgb_encoder.load_state_dict(model_state)
        for epoch in range(self.config["rgb_iters"]):
            translation, quaternion, vertices, texture_maps, lights, tdiff, qdiff = self.rgb_encoder(opt_frames)
            renders = self.rendering(translation, quaternion, vertices, self.encoder.face_features, texture_maps, lights)
            losses_all, losses, jloss = self.rgb_loss_function(renders, segments, input_batch, vertices, texture_maps, tdiff, qdiff)
            if self.best_model["value"] < 0.1 and iters_without_change > 10:
                break
            if epoch < self.config["iterations"] - 1:
                jloss = jloss.mean()
                self.rgb_optimizer.zero_grad()
                jloss.backward()
                self.rgb_optimizer.step()

    def apply_incremental(self, input_batch, segments):
        input_batch, segments = input_batch[None].to(self.device), segments[None].to(self.device)
        for stepi in range(int(self.config["input_frames"]/self.config["inc_step"])):
            if self.config["accumulate"]:
                st = 0
            else:
                st = stepi*self.config["inc_step"]
            en = (stepi+1)*self.config["inc_step"]
            opt_frames = self.keyframes + list(range(st,en))
            breakpoint()
            self.apply(input_batch[:,opt_frames], segments[:,opt_frames][:,:,0,:2], opt_frames)
            self.encoder = self.best_model["encoder"]
            all_parameters = list(self.encoder.parameters())
            self.optimizer = torch.optim.Adam(all_parameters, lr = self.config["learning_rate"])
            self.encoder.train()
            if self.config["write_results"]:
                with torch.no_grad():
                    translation, quaternion, vertices, texture_maps, lights, _, _ = self.encoder(list(range(0,en)))
                    renders = self.rendering(translation, quaternion, vertices, self.encoder.face_features, texture_maps, lights)
                    write_renders(renders, self.write_folder, self.config["inc_step"], en)
                    write_obj_mesh(vertices[0].cpu().numpy(), self.best_model["faces"], self.encoder.face_features[0].cpu().numpy(), os.path.join(self.write_folder,'mesh.obj'))
                    save_image(texture_maps, os.path.join(self.write_folder,'tex.png'))
                    write_video(renders[0,:,0,:3].detach().cpu().numpy().transpose(2,3,1,0), os.path.join(self.write_folder,'im_recon'+'{}.avi'.format(en)), fps=6)
                    write_video(input_batch[0,0:en,:3].cpu().numpy().transpose(2,3,1,0), os.path.join(self.write_folder,'input.avi'), fps=6)
                    write_video((input_batch[0,0:en,:3]*segments[0,0:en,1:2]).cpu().numpy().transpose(2,3,1,0), os.path.join(self.write_folder,'segments.avi'), fps=6)

            if self.config["keyframes"] and not self.config["accumulate"]:
                self.keyframes.append(st)

        return self.best_model