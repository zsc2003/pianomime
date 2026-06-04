# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A task where two shadow hands must play a given MIDI file on a piano."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from dm_control import mjcf
from dm_control.composer import variation as base_variation
from dm_control.composer.observation import observable
from dm_control.mjcf import commit_defaults
from dm_control.utils.rewards import tolerance
from dm_env import specs
from mujoco_utils import collision_utils, spec_utils

import robopianist.models.hands.shadow_hand_constants as hand_consts
from robopianist.models.arenas import stage
from robopianist.music import midi_file
from robopianist.suite import composite_reward
from robopianist.suite.tasks import base
from robopianist.models.piano import piano_constants

# Distance thresholds for the shaping reward.
_FINGER_CLOSE_ENOUGH_TO_KEY = 0.01
_KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05

# Energy penalty coefficient.
_ENERGY_PENALTY_COEF = 5e-3

# Transparency of fingertip geoms.
_FINGERTIP_ALPHA = 1.0

# Bounds for the uniform distribution from which initial hand offset is sampled.
_POSITION_OFFSET = 0.05


class PianoWithShadowHandsResidual(base.PianoTask):
    def __init__(
        self,
        note_trajectory: midi_file.NoteTrajectory = None,
        midi: midi_file.MidiFile = None,
        n_steps_lookahead: int = 1,
        n_seconds_lookahead: Optional[float] = None,
        trim_silence: bool = False,
        wrong_press_termination: bool = False,
        initial_buffer_time: float = 0.0,
        enable_fingering_reward: bool = False,
        enable_forearm_reward: bool = False,
        disable_colorization: bool = False,
        disable_hand_collisions: bool = False,
        augmentations: Optional[Sequence[base_variation.Variation]] = None,
        energy_penalty_coef: float = _ENERGY_PENALTY_COEF,
        randomize_hand_positions: bool = False,
        fingering_lookahead: bool=False,
        midi_start_from: int=0,
        residual_factor: float = 0.02,
        curriculum: bool = False,
        shift: int = 0,
        enable_joints_vel_obs: bool = False,
        smoothness_weight: float = 1.0,
        beta_action: float = 1.0,
        beta_accel: float = 1.0,
        enable_smoothness_reward: bool = False,
        enable_energy_reward: bool = False,
        **kwargs,
    ) -> None:
        """Task constructor.

        Args:
            note_trajectory: A `NoteTrajectory` object.
            midi: A `MidiFile` object.
            n_steps_lookahead: Number of timesteps to look ahead when computing the
                goal state.
            n_seconds_lookahead: Number of seconds to look ahead when computing the
                goal state. If specified, this will override `n_steps_lookahead`.
            trim_silence: If True, shifts the MIDI file so that the first note starts
                at time 0.
            wrong_press_termination: If True, terminates the episode if the hands press
                the wrong keys at any timestep.
            initial_buffer_time: Specifies the duration of silence in seconds to add to
                the beginning of the MIDI file. A non-zero value can be useful for
                giving the agent time to place its hands near the first notes.
            enable_fingering_reward: If True, enables the shaping reward for
                fingering. This will also enable the colorization of the fingertips
                and corresponding keys. Note that if the MIDI file does not contain
                any fingering information, the fingering reward will also be disabled.
            enable_forearm_reward: If True, enables the shaping reward for the
                forearms.
            disable_colorization: If True, disables the colorization of the fingertips
                and corresponding keys.
            disable_hand_collisions: If True, disables collisions between the two hands.
            augmentations: A list of `Variation` objects that will be applied to the
                MIDI file at the beginning of each episode. If None, no augmentations
                will be applied.
            energy_penalty_coef: Coefficient for the energy penalty.
            randomize_hand_positions: If True, randomizes the initial position of the
                hands at the beginning of each episode.
            fingering_lookahead: If True, insert the ahead fingering information in the
                observation.
            midi_start_from: The index of the note to start from.
            residual_factor: The factor to multiply the residual action.
            curriculum: If True, use curriculum learning.
            shift: The number of notes to shift the MIDI file by.
            enable_joints_vel_obs: If True, enable the joint velocity observation.
            smoothness_weight: Weight for the smoothness reward.
            beta_action: Coefficient for action smoothness penalty.
            beta_accel: Coefficient for joint acceleration penalty.
            enable_smoothness_reward: If True, enables the smoothness reward.
        """
        super().__init__(arena=stage.Stage(), **kwargs)
        if note_trajectory is None and midi is None:
            raise ValueError("Either `note_trajectory` or `midi` must be specified.")
        if note_trajectory:
            self._note_traj = note_trajectory
            if trim_silence:
                self._note_traj = self._note_traj.trim_silence()
            if shift:
                self._note_traj = self._note_traj.shift(shift)
            scale = int(0.05 / self.control_timestep)
            self._scale_note_traj(scale)
        else:
            self._midi = midi
            self._initial_midi = midi
            if trim_silence:
                self._midi = self._midi.trim_silence()
        self._n_steps_lookahead = n_steps_lookahead
        if n_seconds_lookahead is not None:
            self._n_steps_lookahead = int(
                np.ceil(n_seconds_lookahead / self.control_timestep)
            )
        self._initial_buffer_time = initial_buffer_time
        self._enable_fingering_reward = enable_fingering_reward
        self._enable_forearm_reward = enable_forearm_reward
        self._wrong_press_termination = wrong_press_termination
        self._disable_colorization = disable_colorization
        self._disable_hand_collisions = disable_hand_collisions
        self._augmentations = augmentations
        self._energy_penalty_coef = energy_penalty_coef
        self._randomize_hand_positions = randomize_hand_positions
        self._fingering_lookahead = fingering_lookahead
        self._midi_start_from = midi_start_from
        self._residual_factor = residual_factor
        self._curriculum = curriculum
        self._curriculum_length = 100 # initial curriculum length (5 seconds)
        self._shift = shift
        self._enable_joints_vel_obs = enable_joints_vel_obs
        self._smoothness_weight = smoothness_weight
        self._beta_action = beta_action
        self._beta_accel = beta_accel
        self._enable_smoothness_reward = enable_smoothness_reward
        self._enable_energy_reward = enable_energy_reward

        if enable_fingering_reward and not disable_colorization:
            self._colorize_fingertips()
        if disable_hand_collisions:
            self._disable_collisions_between_hands()
        self._reset_quantities_at_episode_init()
        self._reset_trajectory()  # Important: call before adding observables.
        self._add_observables()
        self._set_rewards()

    def _set_rewards(self) -> None:
        self._reward_fn = composite_reward.CompositeReward(
            key_press_reward=self._compute_key_press_reward,
            sustain_reward=self._compute_sustain_reward,
        )
        if self._enable_energy_reward:
            self._reward_fn.add("energy_reward", self._compute_energy_reward)
        if self._enable_fingering_reward:
            self._reward_fn.add("fingering_reward", self._compute_fingering_reward)
        if self._enable_forearm_reward:
            self._reward_fn.add("forearm_reward", self._compute_forearm_reward)
        if self._enable_smoothness_reward:
            self._reward_fn.add("smoothness_reward", self._compute_smoothness_reward)

    def _reset_quantities_at_episode_init(self) -> None:
        self._t_idx: int = 0
        self._should_terminate: bool = False
        self._discount: float = 1.0
        # Smoothness reward tracking
        self._current_action = None
        self._prev_action = None
        self._current_qpos = None
        self._prev_qpos = None
        self._prev_prev_qpos = None

    def _maybe_change_midi(self, random_state: np.random.RandomState) -> None:
        if self._augmentations is not None:
            midi = self._initial_midi
            for var in self._augmentations:
                midi = var(initial_value=midi, random_state=random_state)
            self._midi = midi
            self._reset_trajectory()

    def _reset_trajectory(self) -> None:
        if hasattr(self, "_midi"):
            note_traj = midi_file.NoteTrajectory.from_midi(
                self._midi, self.control_timestep
            )
        else:
            note_traj = self._note_traj
        note_traj.add_initial_buffer_time(self._initial_buffer_time)
        self._notes = note_traj.notes[self._midi_start_from:]
        self._sustains = note_traj.sustains[self._midi_start_from:]

    def _scale_note_traj(self, scale):
        # For each note and sustain in note trajectory, repeat it `scale` times.
        # This is used to scale the note trajectory to the same length as the MIDI file.
        new_notes = []
        new_sustains = []
        for note, sustain in zip(self._note_traj.notes, self._note_traj.sustains):
            new_notes += [note] * scale
            new_sustains += [sustain] * scale
        self._note_traj.notes = new_notes
        self._note_traj.sustains = new_sustains

    # Composer methods.

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        # if self._shift:
        #     self._shift_initial_hand_positions(physics, self._shift*piano_constants.WHITE_KEY_WIDTH*7)
        self._maybe_change_midi(random_state)
        self._reset_quantities_at_episode_init()
        self._randomize_initial_hand_positions(physics, random_state)

    def before_step(
        self,
        physics: mjcf.Physics,
        action: np.ndarray,
        random_state: np.random.RandomState,
    ) -> None:
        """Applies the control to the hands and the sustain pedal to the piano."""
        # Track action for smoothness reward
        self._prev_action = self._current_action.copy() if self._current_action is not None else None
        self._current_action = action.copy()
        action_right, action_left = np.split(action[:-1], 2)
        action_right[-3] += piano_constants.WHITE_KEY_WIDTH * self._shift * 7
        action_left[-3] += piano_constants.WHITE_KEY_WIDTH * self._shift * 7
        self.right_hand.apply_action(physics, action_right, random_state)
        self.left_hand.apply_action(physics, action_left, random_state)
        action[-1] = self._goal_state[0][-1]
        self.piano.apply_sustain(physics, action[-1], random_state)

    def after_step(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.
        self._t_idx += 1
        # Track joint positions for smoothness reward
        lh_qpos = self.left_hand.observables.joints_pos(physics).copy()
        rh_qpos = self.right_hand.observables.joints_pos(physics).copy()
        current_qpos = np.concatenate([lh_qpos, rh_qpos])
        self._prev_prev_qpos = self._prev_qpos.copy() if self._prev_qpos is not None else None
        self._prev_qpos = self._current_qpos.copy() if self._current_qpos is not None else None
        self._current_qpos = current_qpos
        self._should_terminate = (self._t_idx - 1) == len(self._notes) - 1
        if self._curriculum:
            self._should_terminate = self._should_terminate or self._t_idx == self._curriculum_length
        self._goal_current = self._goal_state[0]
        if self._enable_fingering_reward:
            self._rh_keys_current = self._rh_keys
            self._lh_keys_current = self._lh_keys
            if not self._disable_colorization:
                self._colorize_keys_without_fingering(physics)
        should_not_be_pressed = np.flatnonzero(1 - self._goal_current[:-1])
        self._failure_termination = self.piano.activation[should_not_be_pressed].any()

    def extend_curriculum(self) -> None:
        self._curriculum_length += 100

    def get_reward(self, physics: mjcf.Physics) -> float:
        return self._reward_fn.compute(physics)

    def get_discount(self, physics: mjcf.Physics) -> float:
        del physics  # Unused.
        return self._discount

    def should_terminate_episode(self, physics: mjcf.Physics) -> bool:
        del physics  # Unused.
        if self._should_terminate:
            return True
        if self._wrong_press_termination and self._failure_termination:
            self._discount = 0.0
            return True
        return False

    @property
    def task_observables(self):
        return self._task_observables

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        right_spec = self.right_hand.action_spec(physics)
        left_spec = self.left_hand.action_spec(physics)
        hands_spec = spec_utils.merge_specs([right_spec, left_spec])
        interval = hands_spec.maximum - hands_spec.minimum
        if self._residual_factor == 1:
            minimum = hands_spec.minimum
            maximum = hands_spec.maximum
        else:
            minimum = -interval * self._residual_factor
            maximum = interval * self._residual_factor
        # Append 0 and 1 for sustain pedal.
        minimum = np.append(minimum, 0.0)
        maximum = np.append(maximum, 1.0)
        action_spec = specs.BoundedArray(
            shape=(47,),
            dtype='float32',
            minimum=minimum,
            maximum=maximum,
            name="residual",
        )
        return action_spec

    # Other.

    @property
    def midi(self) -> midi_file.MidiFile:
        return self._midi

    @property
    def reward_fn(self) -> composite_reward.CompositeReward:
        return self._reward_fn

    # Helper methods.

    def _compute_forearm_reward(self, physics: mjcf.Physics) -> float:
        """Reward for not colliding the forearms."""
        if collision_utils.has_collision(
            physics,
            [g.full_identifier for g in self.right_hand.root_body.geom],
            [g.full_identifier for g in self.left_hand.root_body.geom],
        ):
            return 0.0
        return 0.5

    def _compute_sustain_reward(self, physics: mjcf.Physics) -> float:
        """Reward for pressing the sustain pedal at the right time."""
        del physics  # Unused.
        return tolerance(
            self._goal_current[-1] - self.piano.sustain_activation[0],
            bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
            margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
            sigmoid="gaussian",
        )
        # return 0

    def _compute_energy_reward(self, physics: mjcf.Physics) -> float:
        """Reward for minimizing energy."""
        rew = 0.0
        for hand in [self.right_hand, self.left_hand]:
            power = hand.observables.actuators_power(physics).copy()
            rew -= self._energy_penalty_coef * np.sum(power)
        return rew

    def _compute_key_press_reward(self, physics: mjcf.Physics) -> float:
        """Reward for pressing the right keys at the right time."""
        del physics  # Unused.
        on = np.flatnonzero(self._goal_current[:-1])
        rew = 0.0
        # It's possible we have no keys to press at this timestep, so we need to check
        # that `on` is not empty.
        if on.size > 0:
            actual = np.array(self.piano.state / self.piano._qpos_range[:, 1])
            rews = tolerance(
                self._goal_current[:-1][on] - actual[on],
                bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
                margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
                sigmoid="gaussian",
            )
            rew += 0.5 * rews.mean()
        # If there are any false positives, the remaining 0.5 reward is lost.
        off = np.flatnonzero(1 - self._goal_current[:-1])
        rew += 0.5 * (1 - float(self.piano.activation[off].any()))
        return 2*rew

    def _compute_fingering_reward(self, physics: mjcf.Physics) -> float:
        """Reward for minimizing the distance between the fingers and the keys."""

        def _distance_finger_to_key(
            hand_keys: List[Tuple[int, int]], hand
        ) -> List[float]:
            distances = []
            for key, mjcf_fingering in hand_keys:
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                fingertip_pos = physics.bind(fingertip_site).xpos.copy()
                key_geom = self.piano.keys[key].geom[0]
                key_geom_pos = physics.bind(key_geom).xpos.copy()
                key_geom_pos[-1] += 0.5 * physics.bind(key_geom).size[2]
                key_geom_pos[0] += 0.35 * physics.bind(key_geom).size[0]
                diff = key_geom_pos - fingertip_pos
                distances.append(float(np.linalg.norm(diff)))
            return distances

        distances = _distance_finger_to_key(self._rh_keys_current, self.right_hand)
        distances += _distance_finger_to_key(self._lh_keys_current, self.left_hand)

        # Case where there are no keys to press at this timestep.
        if not distances:
            return 0.0

        rews = tolerance(
            np.hstack(distances),
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        return float(np.mean(rews))

    def _compute_smoothness_reward(self, physics: mjcf.Physics) -> float:
        """Reward for smooth actions and joint trajectories.
        
        r_smooth = exp(-beta_action * ||a_t[0:46] - a_{t-1}[0:46]||^2 
                       - beta_accel * ||q_t - 2*q_{t-1} + q_{t-2}||^2)
        
        Returns 0.0 for the first 2 timesteps (insufficient history).
        """
        del physics  # Joint positions already tracked in after_step.
        
        # Action rate penalty: ||a_t[0:46] - a_{t-1}[0:46]||^2 (exclude sustain pedal)
        if self._current_action is None or self._prev_action is None:
            return 0.0
        
        assert self._current_action.shape == (47,), (
            f"Expected action shape (47,), got {self._current_action.shape}"
        )
        action_diff = self._current_action[:-1] - self._prev_action[:-1]  # Exclude sustain
        action_rate = np.sum(action_diff ** 2)
        
        # Acceleration penalty: ||q_t - 2*q_{t-1} + q_{t-2}||^2
        if self._current_qpos is None or self._prev_qpos is None or self._prev_prev_qpos is None:
            accel_penalty = 0.0
        else:
            assert self._current_qpos.shape == (54,), (
                f"Expected qpos shape (54,), got {self._current_qpos.shape}"
            )
            accel = self._current_qpos - 2.0 * self._prev_qpos + self._prev_prev_qpos
            accel_penalty = np.sum(accel ** 2)
        
        smoothness = np.exp(
            -self._beta_action * action_rate - self._beta_accel * accel_penalty
        )
        
        return self._smoothness_weight * smoothness

    def _update_goal_state(self) -> None:
        # Observable callables get called after `after_step` but before
        # `should_terminate_episode`. Since we increment `self._t_idx` in `after_step`,
        # we need to guard against out of bounds indexing. Note that the goal state
        # does not matter at this point since we are terminating the episode and this
        # update is usually meant for the next timestep.
        if self._t_idx == len(self._notes):
            return

        self._goal_state = np.zeros(
            (self._n_steps_lookahead + 1, self.piano.n_keys + 1),
            dtype=np.float64,
        )
        t_start = self._t_idx
        t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.key for note in self._notes[t]]
            self._goal_state[i, keys] = 1.0
            self._goal_state[i, -1] = self._sustains[t]

    def get_next_goal_state(self) -> np.ndarray:
        # Observable callables get called after `after_step` but before
        # `should_terminate_episode`. Since we increment `self._t_idx` in `after_step`,
        # we need to guard against out of bounds indexing. Note that the goal state
        # does not matter at this point since we are terminating the episode and this
        # update is usually meant for the next timestep.
        t_idx = self._t_idx + 1
        if t_idx == len(self._notes):
            return self._goal_state

        goal_state = np.zeros(
            (self._n_steps_lookahead + 1, self.piano.n_keys + 1),
            dtype=np.float64,
        )
        t_start = t_idx
        t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.key for note in self._notes[t]]
            goal_state[i, keys] = 1.0
            goal_state[i, -1] = self._sustains[t]
        return goal_state

    def _update_fingering_state(self) -> None:
        if self._t_idx == len(self._notes):
            return
        fingering = [note.fingering for note in self._notes[self._t_idx]]
        fingering_keys = [note.key for note in self._notes[self._t_idx]]
        # print("fingering", fingering) # from the rightest (right hand) to the leftest
        # print("keys", fingering_keys) # from the leftest to the rightest

        # Split fingering into right and left hand.
        self._rh_keys: List[Tuple[int, int]] = []
        self._lh_keys: List[Tuple[int, int]] = []
        for key, finger in enumerate(fingering):
            piano_key = fingering_keys[key]
            if finger < 5:
                self._rh_keys.append((piano_key, finger))
            else:
                self._lh_keys.append((piano_key, finger - 5))

        if not self._fingering_lookahead:
            # For each hand, set the finger to 1 if it is used and 0 otherwise.
            self._fingering_state = np.zeros((2, 5), dtype=np.float64)
            for hand, keys in enumerate([self._rh_keys, self._lh_keys]):
                for key, mjcf_fingering in keys:
                    self._fingering_state[hand, mjcf_fingering] = 1.0
            # print(self._fingering_state)
        else:
            self._fingering_state = np.zeros((self._n_steps_lookahead + 1, 10), dtype=np.float64)
            t_start = self._t_idx
            t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
            for i, t in enumerate(range(t_start, t_end)):
                fingers = [note.fingering for note in self._notes[t]]
                self._fingering_state[i, fingers] = 1.0
            # print(self._fingering_state)

    def _add_observables(self) -> None:
        # Enable hand observables.
        enabled_observables = [
            "joints_pos",
            # NOTE(kevin): This observable was previously enabled but it is redundant
            # since it is encoded in the joint positions, specifically via the forearm
            # slider joints (which are in units of meters).
            # "position",
        ]
        if self._enable_joints_vel_obs:
            enabled_observables += ["joints_vel"]
        for hand in [self.right_hand, self.left_hand]:
            for obs in enabled_observables:
                getattr(hand.observables, obs).enabled = True

        # This returns the current state of the piano keys.
        self.piano.observables.state.enabled = True
        self.piano.observables.sustain_state.enabled = True

        # This returns the goal state for the current timestep and n steps ahead.
        def _get_goal_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_goal_state()
            return self._goal_state.ravel()

        goal_observable = observable.Generic(_get_goal_state)
        goal_observable.enabled = True
        self._task_observables = {"goal": goal_observable}

        # This adds fingering information for the current timestep.
        def _get_fingering_state(physics) -> np.ndarray:
            del physics  # Unused.
            self._update_fingering_state()
            return self._fingering_state.ravel()

        fingering_observable = observable.Generic(_get_fingering_state)
        fingering_observable.enabled = self._enable_fingering_reward
        self._task_observables["fingering"] = fingering_observable

    def _colorize_fingertips(self) -> None:
        """Colorize the fingertips of the hands."""
        for hand in [self.right_hand, self.left_hand]:
            for i, body in enumerate(hand.fingertip_bodies):
                color = hand_consts.FINGERTIP_COLORS[i] + (_FINGERTIP_ALPHA,)
                for geom in body.find_all("geom"):
                    if geom.dclass.dclass == "plastic_visual":
                        geom.rgba = color
                # Also color the fingertip sites.
                hand.fingertip_sites[i].rgba = color

    def _colorize_keys_without_fingering(self, physics) -> None:
        """Colorize the correctly or wrongly pressed keys."""
        activation = self.piano.activation
        active_key_indices = np.where(activation == True)[0]
        key_should_pressed = []
        if len(self._rh_keys_current) > 0:
            key_should_pressed += [key for key, _ in self._rh_keys_current]
        if len(self._lh_keys_current) > 0:
            key_should_pressed += [key for key, _ in self._lh_keys_current]
        for key in active_key_indices:
            if key in key_should_pressed:
                # Correctly pressed keys
                key_geom = self.piano.keys[key].geom[0]
                # Green
                physics.bind(key_geom).rgba = (0.0, 1.0, 0.0, 1.0)
            else:
                # Wrongly pressed keys
                key_geom = self.piano.keys[key].geom[0]
                # Red
                physics.bind(key_geom).rgba = (1.0, 0.0, 0.0, 1.0)
        for key in key_should_pressed:
            if key not in active_key_indices:
                # Unpressed keys
                key_geom = self.piano.keys[key].geom[0]
                # Yellow
                physics.bind(key_geom).rgba = (1.0, 1.0, 0.0, 1.0)

    def _colorize_keys(self, physics) -> None:
        """Colorize the keys by the corresponding fingertip color."""
        for hand, keys in zip(
            [self.right_hand, self.left_hand],
            [self._rh_keys_current, self._lh_keys_current],
        ):
            for key, mjcf_fingering in keys:
                key_geom = self.piano.keys[key].geom[0]
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                if not self.piano.activation[key]:
                    physics.bind(key_geom).rgba = tuple(fingertip_site.rgba[:3]) + (
                        1.0,
                    )

    def _disable_collisions_between_hands(self) -> None:
        """Disable collisions between the hands."""
        for hand in [self.right_hand, self.left_hand]:
            for geom in hand.mjcf_model.find_all("geom"):
                # If both hands have the same contype and conaffinity, then they can't
                # collide. They can still collide with the piano since the piano has
                # contype 0 and conaffinity 1. Lastly, we make sure we're not changing
                # the contype and conaffinity of the hand geoms that are already
                # disabled (i.e., the visual geoms).
                commit_defaults(geom, ["contype", "conaffinity"])
                if geom.contype == 0 and geom.conaffinity == 0:
                    continue
                geom.conaffinity = 0
                geom.contype = 1

    def _randomize_initial_hand_positions(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        """Randomize the initial position of the hands."""
        if not self._randomize_hand_positions:
            return
        offset = random_state.uniform(low=-_POSITION_OFFSET, high=_POSITION_OFFSET)
        for hand in [self.right_hand, self.left_hand]:
            hand.shift_pose(physics, (0, offset, 0))

    def _shift_initial_hand_positions(
        self, physics: mjcf.Physics, shift: float
    ) -> None:
        """Shift the initial position of the hands."""
        for hand in [self.right_hand, self.left_hand]:
            hand.shift_pose(physics, (0, shift, 0))

    def get_fingertip_pos(self, physics: mjcf.Physics) -> np.ndarray:
        lh_wrist = physics.named.data.site_xpos["lh_shadow_hand/wrist_site"].flatten()
        rh_wrist = physics.named.data.site_xpos["rh_shadow_hand/wrist_site"].flatten()
        lh_fingertips = np.array(physics.bind(self.left_hand.fingertip_sites).xpos).flatten()
        rh_fingertips = np.array(physics.bind(self.right_hand.fingertip_sites).xpos).flatten()
        lh_current = np.concatenate((lh_wrist, lh_fingertips))
        rh_current = np.concatenate((rh_wrist, rh_fingertips))
        return lh_current, rh_current
