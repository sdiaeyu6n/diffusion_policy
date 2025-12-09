from typing import Dict, Callable, Tuple
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray],
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')

        if type == 'rgb':
            # --- NEW: map expected camera keys to actual env_obs keys ---
            src_key = key
            if src_key not in env_obs:
                # Lite6 setup: env has 'camera_0' and 'camera_1'
                # but config/checkpoint expects 'camera_1' and 'camera_3'
                if src_key == 'camera_1' and 'camera_0' in env_obs:
                    src_key = 'camera_0'
                elif src_key == 'camera_3' and 'camera_1' in env_obs:
                    src_key = 'camera_1'
                else:
                    raise KeyError(
                        f"Camera key {key!r} not found in env_obs. "
                        f"Available keys: {list(env_obs.keys())}"
                    )

            this_imgs_in = env_obs[src_key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi, hi),
                    output_res=(wo, ho),
                    bgr_to_rgb=False
                )
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1)

        elif type == 'low_dim':
            this_data_in = env_obs[key]
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[..., [0, 1]]
            obs_dict_np[key] = this_data_in

    return obs_dict_np


def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co, ho, wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
