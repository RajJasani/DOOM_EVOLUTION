# Member 2 Handoff: Spatial Intelligence & Perception

**Target Role:** Member 2 (Spatial Intelligence & Perception)  
**Lead Coordinator:** Member 1 (AI Architecture & Integration)  
**Target Module:** `perception/`  
**Reference Specs:** `docs/CONTRACT.md`, `INTERFACES.md`  

---

## 1. Current Implementation Status

An operational perception foundation has been established in `perception/spatial.py` and validated via `tests/test_perception.py`:
- **`SpatialPerception.process_observation(obs)`**: Validates raw observation dictionary inputs and formats them into sanitized NumPy arrays.
- **`SpatialPerception.detect_doors(ray_dist, ray_hit)`**: Accurately detects normal doors (hit type 2) and locked doors (hit type 3), indexing ray directions and closest distances.
- **`SpatialPerception.get_forward_clearance(ray_dist, window_rays=5)`**: Computes forward hallway clearance across frontal ray cones.
- **`SpatialPerception.get_nearest_threat(ent_feats, ent_mask)`**: Extracts distance, relative angle, and ego-velocities of the closest hostile entity.

All 4 initial tests in `tests/test_perception.py` pass in 0.03 seconds.

---

## 2. Architectural Relationship with Member 3 & Member 1

- **Current Direct Flow:** `submission/agent.py` routes raw observation dictionaries directly into `DoomNetwork.get_action(obs, hx)` during evaluation. `DoomNetwork` internally embeds the 64 rays, 32 visible entities, and 17 vitals directly via circular CNN and multi-head attention.
- **Member 2 Primary Responsibility:**
  1. Provide spatial analysis and hazard/door extraction for **Member 4's reward shaping and curriculum design**.
  2. Provide spatial feature refinement without adding latency bottlenecks to the 40 ms/step inference budget.
  3. *(Optional)* If the team formally approves visual feature processing, provide a compact RGB/Depth CNN encoder outputting to key `"visual_embedding"`.

---

## 3. Exact Function Signatures & Data Contracts

### 3.1 Observation Normalization (`perception/spatial.py`)
```python
def process_observation(self, obs: Mapping[str, Any]) -> dict[str, Any]:
```
- **Input:**
  - `obs["vitals"]`: Array `(17,)`, `float32`.
  - `obs["rays"]["distance"]`: Array `(64,)`, `float32`, values $\in [0.0, 1.0]$.
  - `obs["rays"]["hit_type"]`: Array `(64,)`, `int64`, values $\in \{0, 1, 2, 3, 4, 5\}$.
  - `obs["visible_entities"]["features"]`: Array `(32, 5)`, `float32`.
  - `obs["visible_entities"]["entity_type"]`: Array `(32,)`, `int64`.
  - `obs["visible_entities"]["mask"]`: Array `(32,)`, `float32` in $\{0.0, 1.0\}$.
- **Output:**
  - Dict containing sanitized, type-checked NumPy arrays matching the above shapes.

### 3.2 Door & Lock Detection
```python
def detect_doors(self, ray_dist: np.ndarray, ray_hit: np.ndarray) -> dict[str, Any]:
```
- **Inputs:** `ray_dist: (64,) float32`, `ray_hit: (64,) int64`.
- **Output:**
  ```python
  {
      "has_door": bool,
      "has_locked_door": bool,
      "door_dist": float,            # In [0.0, 1.0] (1.0 = no door)
      "door_ray_idx": int | None,    # Index 0..63 in player frame
      "locked_door_dist": float,
      "locked_door_ray_idx": int | None,
  }
  ```

### 3.3 Kinematic Threat Parsing
```python
def get_nearest_threat(self, ent_feats: np.ndarray, ent_mask: np.ndarray) -> dict[str, float] | None:
```
- **Inputs:** `ent_feats: (32, 5) float32`, `ent_mask: (32,) float32`.
- **Output:** `None` if `ent_mask.sum() == 0`, else:
  ```python
  {
      "norm_distance": float,        # dist / 1024.0
      "sin_angle": float,            # Positive = left, 0 = forward
      "cos_angle": float,            # Positive = forward
      "rel_vel_fwd": float,
      "rel_vel_right": float,
  }
  ```

---

## 4. Pending Deliverables & Tasks for Member 2

1. **Pickup Item Spatial Detector:**
   - Add `detect_pickups(ray_dist, ray_hit)` to identify ammo/health/key items (hit type 5) to assist Member 4 with exploration incentives.
2. **Door Clearance Verification:**
   - Implement `is_door_traversable(ray_dist, ray_hit, door_ray_idx)` to determine when an opened door allows passage (rays transition from hit type 2 to passing through).
3. **Optional Vision Encoder (Only if Approved):**
   - If vision is deemed necessary, create `perception/vision.py` taking `(B, 4, 120, 160)` (RGB + Depth) and producing `(B, 64)`.
   - **Hard Constraint:** Inference latency must not exceed 2.0 ms on 1 CPU thread.
4. **Unit Test Expansion:**
   - Expand `tests/test_perception.py` to cover corrupted shapes, extreme ranges, and 100% test branch coverage.
