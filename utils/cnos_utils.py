if './repositories/cnos' not in sys.path:
    sys.path.append('./repositories/cnos')

def get_default_detections_per_scene_and_image(default_detections_file: Path) -> Dict[Tuple[int, int], List]:

    from src.utils.bbox_utils import xywh_to_xyxy
    with open(default_detections_file, 'r') as f:
        default_detections_data = json.load(f)
        default_detections_scene_im_dict = defaultdict(list)
        for i, item in enumerate(default_detections_data):
            im_id: int = item['image_id']
            scene_id: int = item['scene_id']
            default_detections_scene_im_dict[(im_id, scene_id)].append(item)

    default_detections_scene_im_dict_cnos_format = {}
    for key, item in default_detections_scene_im_dict.items():
        rle_to_mask(detection)
        detections_dist = {
            'object_ids': None,
            'bbox': None,
            'scores': None,
            'masks': None,
        }
        [i for i in range(len(item))]
        xywh_to_xyxy
        breakpoint()
    return default_detections_scene_im_dict


Detection = namedtuple('Detection', ['object_id', 'segmentation_mask', 'score'])


def get_default_detections_for_image(default_detections_data_scene_im_dict: Dict[Tuple[int, int], List], scene_id: int,
                                     im_id: int, device: str = 'cpu') -> List[Detection]:
    detections_for_image = []  # Initialize as list, not dict
    for detections_data in default_detections_data_scene_im_dict[(im_id, scene_id)]:
        segmentation_rle_format = detections_data['segmentation']

        mask = decode_rle_list(segmentation_rle_format)
        mask_tensor = torch.tensor(mask, device=device)
        detections_data['segmentation_tensor'] = mask_tensor

        detections_for_image.append(detections_data)

    detections_for_image.sort(key=lambda x: (x['score'], x['category_id']), reverse=True)

    breakpoint()
    sorted_detections = [Detection(object_id=detection['category_id'],
                                   segmentation_mask=detection['segmentation_tensor'],
                                   score=detection['score'])
                         for detection in detections_for_image]

    return sorted_detections