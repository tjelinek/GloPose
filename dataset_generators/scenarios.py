from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MovementScenario:
    steps: int = 0
    initial_rotation: np.ndarray = np.array([0.01, 0.01, 0.01])
    initial_translation: np.ndarray = np.array([0.0, 0.0, 0.0])
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


def generate_zero_rotations(steps=72) -> MovementScenario:
    rotations_x = np.zeros(steps)
    rotations_y = np.zeros(rotations_x.shape)
    rotations_z = np.zeros(rotations_x.shape)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_x(step=10.0) -> MovementScenario:
    rotations_x = np.arange(0.0, 1 * 360.0, step)
    rotations_y = np.zeros(rotations_x.shape)
    rotations_z = np.zeros(rotations_x.shape)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_y(step=10.0) -> MovementScenario:
    rotations_y = np.arange(0.0, 1 * 360.0, step)
    rotations_x = np.zeros(rotations_y.shape)
    rotations_z = np.zeros(rotations_x.shape)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_z(step=10.0) -> MovementScenario:
    rotations_z = np.arange(0.0, 1 * 360.0, step)
    rotations_x = np.zeros(rotations_z.shape)
    rotations_y = np.zeros(rotations_x.shape)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_xy(step=10.0) -> MovementScenario:
    rotations_x = np.arange(0.0, 1 * 360.0, step)
    rotations_y = np.arange(0.0, 1 * 360.0, step)
    rotations_z = np.zeros(rotations_x.shape)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_xz(step=10.0) -> MovementScenario:
    rotations_x = np.arange(0.0, 1 * 360.0, step)
    rotations_y = np.zeros(rotations_x.shape)
    rotations_z = np.arange(0.0, 1 * 360.0, step)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_yz(step=10.0) -> MovementScenario:
    rotations_x = np.zeros(np.arange(0.0, 1 * 360.0, step).shape)
    rotations_y = np.arange(0.0, 1 * 360.0, step)
    rotations_z = np.arange(0.0, 1 * 360.0, step)

    scenario = MovementScenario(
        rotations=[np.array([x, y, z]) for x, y, z in zip(rotations_x, rotations_y, rotations_z)])
    return scenario


def generate_rotations_xyz(step=10.0) -> MovementScenario:
    rotations_x = np.arange(0.0, 1 * 360.0, step)
    rotations_y = np.arange(0.0, 1 * 360.0, step)
    rotations_z = np.arange(0.0, 1 * 360.0, step)

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
