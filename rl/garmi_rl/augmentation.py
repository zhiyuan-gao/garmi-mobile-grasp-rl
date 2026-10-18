"""GA-DDPG-style expert perturbations and explicit supervised-label provenance."""
import numpy as np
from scipy.spatial.transform import Rotation


def sample_hand_perturbation(position, quaternion_wxyz, rng):
    """Official random_perturb ranges, composed in the current hand frame.

    GA-DDPG 14a65e1 env/panda_scene.py: translation per axis +/- .04 m,
    Euler rotation per axis +/- .2 rad. GARMI executes this target through
    its existing Giskard controller; this is not an instantaneous reset.
    """
    translation = rng.uniform(-.04, .04, 3)
    angles = rng.uniform(-.2, .2, 3)
    rotation = Rotation.from_quat(np.asarray(quaternion_wxyz)[[1, 2, 3, 0]])
    target_rotation = rotation * Rotation.from_euler('xyz', angles)
    return dict(pose=np.r_[np.asarray(position) + rotation.apply(translation),
                          target_rotation.as_quat()[[3, 0, 1, 2]]].tolist(),
                translation_hand_m=translation.tolist(), rotation_hand_rad=angles.tolist())


def approach_label_mask(phases, perturbation_mask=None):
    """Perturbations produce states, never correct-action supervision."""
    phases = np.asarray(phases)
    mask = (phases[:-1] == 0) & (phases[1:] == 0)
    if perturbation_mask is not None:
        noise = np.asarray(perturbation_mask)
        assert noise.dtype == np.bool_ and noise.shape == mask.shape
        mask &= ~noise
    return mask
