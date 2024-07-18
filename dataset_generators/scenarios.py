from dataclasses import dataclass, asdict, field

import torch
from kornia.geometry.conversions import axis_angle_to_quaternion
from typing import List

import numpy as np

def default_initial_rotation():
    return np.array([0., 0., 0.])


def default_initial_translation():
    return np.array([0., 0., 0.])


@dataclass
class MovementScenario:
    steps: int = 0
    initial_rotation: np.ndarray = field(default_factory=default_initial_rotation)
    initial_translation: np.ndarray = field(default_factory=default_initial_translation)
    rotations: List[np.ndarray] = None
    translations: List[np.ndarray] = None

    def __post_init__(self):
        if self.rotations is None and self.translations is None:
            self.rotations = [np.array([0.0, 0.0, 0.0])]
            self.translations = [np.array([0.0, 0.0, 0.0])]

        elif self.rotations is None:
            self.rotations = [np.array([0.0, 0.0, 0.0])] * len(self.translations)

        elif self.translations is None:
            self.translations = [np.array([0.0, 0.0, 0.0])] * len(self.rotations)

        self.steps = len(self.translations)

    @property
    def rotation_quaternions(self) -> List[np.ndarray]:
        quaternions = []

        for rot_deg in self.rotations:
            rotations_radians = np.deg2rad(rot_deg)
            rotation_quaternion = axis_angle_to_quaternion(torch.from_numpy(rotations_radians)).numpy()
            quaternions.append(rotation_quaternion)

        return quaternions

    @property
    def rotation_axis_angles(self) -> np.ndarray:
        return np.deg2rad(np.asarray(self.rotations))

    def get_dict(self):
        scenario_dict = asdict(self)

        scenario_dict['rotation_quaternions'] = self.rotation_quaternions

        return scenario_dict


def get_full_rotation(step):
    eps = 1e-10
    return np.arange(0.0, 1 * 360.0 + eps, step)


def generate_zero_rotations(steps=72) -> MovementScenario:
    rotations_x = np.zeros(steps)
    rotations_y = np.zeros(rotations_x.shape)
    rotations_z = np.zeros(rotations_x.shape)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_x(step=10.0) -> MovementScenario:
    rotations_x = get_full_rotation(step)
    rotations_y = np.zeros(rotations_x.shape)
    rotations_z = np.zeros(rotations_x.shape)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_y(step=10.0) -> MovementScenario:
    rotations_y = get_full_rotation(step)
    rotations_x = np.zeros(rotations_y.shape)
    rotations_z = np.zeros(rotations_x.shape)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_z(step=10.0) -> MovementScenario:
    rotations_z = get_full_rotation(step)
    rotations_x = np.zeros(rotations_z.shape)
    rotations_y = np.zeros(rotations_x.shape)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_xy(step=10.0) -> MovementScenario:
    rotations_x = get_full_rotation(step)
    rotations_y = get_full_rotation(step)
    rotations_z = np.zeros(rotations_x.shape)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_xz(step=10.0) -> MovementScenario:
    rotations_x = get_full_rotation(step)
    rotations_y = np.zeros(rotations_x.shape)
    rotations_z = get_full_rotation(step)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_yz(step=10.0) -> MovementScenario:
    rotations_x = np.zeros(get_full_rotation(step).shape)
    rotations_y = get_full_rotation(step)
    rotations_z = get_full_rotation(step)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_xyz(step=10.0) -> MovementScenario:
    rotations_x = get_full_rotation(step)
    rotations_y = get_full_rotation(step)
    rotations_z = get_full_rotation(step)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_sinusoidal_translations(steps=72) -> MovementScenario:
    step = steps_to_periodic_linspace(steps)
    x = np.linspace(0, (steps - 1) * step, steps)
    translations_x = np.sin(x) * 0.5
    translations_y = np.zeros(translations_x.shape)
    translations_z = np.zeros(translations_x.shape)

    result_tuples = list(zip(translations_x, translations_y, translations_z))
    result = MovementScenario(translations=[np.array([x, y, z]) for x, y, z in result_tuples])
    return result


def generate_circular_translation(steps=72) -> MovementScenario:
    x = steps_to_periodic_linspace(steps)
    translations_x = np.cos(x) * 0.5
    translations_y = np.sin(x) * 0.5
    translations_z = np.zeros(translations_x.shape)

    result_tuples = list(zip(translations_x, translations_y, translations_z))
    translations = [np.array([x, y, z]) for x, y, z in result_tuples]
    initial_translation = np.array([0.0, 0.0, 0.0])
    result = MovementScenario(translations=translations, initial_translation=initial_translation)
    return result


def generate_in_depth_translations(steps=72) -> MovementScenario:
    x = steps_to_periodic_linspace(steps)
    translations_y = np.zeros(x.shape)
    translations_x = np.sin(x) * 1
    translations_z = np.cos(x) * 4

    result_tuples = list(zip(translations_x, translations_y, translations_z))
    translations = [np.array([x, y, z]) for x, y, z in result_tuples]
    initial_translation = np.array([0.0, 0.0, -4.0])
    result = MovementScenario(translations=translations, initial_translation=initial_translation)
    return result


def generate_translation_that_is_off(steps=72) -> MovementScenario:
    step = steps_to_periodic_linspace(steps)
    x = np.linspace(0, (steps - 1) * step, steps) * 0
    translations_x = np.cos(x)
    translations_y = np.sin(x)
    translations_z = np.zeros(translations_x.shape)

    result_tuples = list(zip(translations_x, translations_y, translations_z))
    result = MovementScenario(translations=[np.array([x, y, z]) for x, y, z in result_tuples])
    return result


def steps_to_periodic_linspace(steps):
    return np.linspace(0, 2 * np.pi, steps + 1)[:-1]
