import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

NUM_VERTICES = 8
NUM_EDGES = 12
vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
edges = ti.Vector.field(2, dtype=ti.i32, shape=NUM_EDGES)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)

# ── 新增：Python 端四元数工具 ──────────────────────────────
def euler_to_quat(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx/2), np.sin(rx/2)
    cy, sy = np.cos(ry/2), np.sin(ry/2)
    cz, sz = np.cos(rz/2), np.sin(rz/2)
    return np.array([
        cz*cy*cx + sz*sy*sx,   # w
        cz*cy*sx - sz*sy*cx,   # x
        cz*sy*cx + sz*cy*sx,   # y
        sz*cy*cx - cz*sy*sx,   # z
    ])

def slerp(qa, qb, t):
    dot = np.dot(qa, qb)
    if dot < 0:                    # 取最短弧
        qb, dot = -qb, -dot
    if dot > 0.9995:               # 退化为 LERP 防止数值问题
        q = qa + t * (qb - qa)
        return q / np.linalg.norm(q)
    theta0 = np.arccos(np.clip(dot, -1, 1))
    theta  = theta0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / np.sin(theta0)
    s1 = np.sin(theta) / np.sin(theta0)
    return s0 * qa + s1 * qb

def quat_to_angles(q):
    """四元数 → 欧拉角（度），顺序 Rz*Ry*Rx，供原有 get_rotation_matrix 使用"""
    w, x, y, z = q / np.linalg.norm(q)
    rx = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    ry = np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))
    rz = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return ti.Vector([float(rx), float(ry), float(rz)])
# ────────────────────────────────────────────────────────────

@ti.func
def get_rotation_matrix(angles: ti.template()):
    rad = angles * ti.math.pi / 180.0
    cx, sx = ti.cos(rad[0]), ti.sin(rad[0])
    cy, sy = ti.cos(rad[1]), ti.sin(rad[1])
    cz, sz = ti.cos(rad[2]), ti.sin(rad[2])
    rx = ti.Matrix([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]], dt=ti.f32)
    ry = ti.Matrix([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dt=ti.f32)
    rz = ti.Matrix([[cz,-sz,0,0],[sz,cz,0,0],[0,0,1,0],[0,0,0,1]], dt=ti.f32)
    return rz @ ry @ rx

@ti.kernel
def compute_transform(angles: ti.types.vector(3, ti.f32)):
    model = get_rotation_matrix(angles)
    view  = ti.Matrix([[1,0,0,0],[0,1,0,0],[0,0,1,-5],[0,0,0,1]], dt=ti.f32)
    f, n, aspect, fov = 50.0, 0.1, 1.0, 45.0
    tan_half = ti.tan(fov * ti.math.pi / 360.0)
    proj = ti.Matrix.zero(ti.f32, 4, 4)
    proj[0,0], proj[1,1] = 1/(aspect*tan_half), 1/tan_half
    proj[2,2], proj[2,3] = -(f+n)/(f-n), -(2*f*n)/(f-n)
    proj[3,2] = -1.0
    mvp = proj @ view @ model
    for i in range(NUM_VERTICES):
        v4     = ti.Vector([vertices[i][0], vertices[i][1], vertices[i][2], 1.0])
        v_clip = mvp @ v4
        v_ndc  = ti.Vector([v_clip[0]/v_clip[3], v_clip[1]/v_clip[3]])
        screen_coords[i] = (v_ndc + 1.0) / 2.0

def init_cube():
    v_list = [[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1]]
    for i in range(8): vertices[i] = v_list[i]
    e_list = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    for i in range(12): edges[i] = e_list[i]

def draw_cube(gui, color):
    coords = screen_coords.to_numpy()
    edge_indices = edges.to_numpy()
    for i in range(NUM_EDGES):
        gui.line(coords[edge_indices[i][0]], coords[edge_indices[i][1]], radius=2, color=color)

def main():
    init_cube()
    gui = ti.GUI("Rotation Interpolation Rt", res=(700, 700))

    pose_r0 = [20.0, -30.0, 10.0]
    pose_r1 = [-20.0, 60.0, -45.0]

    # ── 只改这里：欧拉角 → 四元数 ──
    qa = euler_to_quat(*pose_r0)
    qb = euler_to_quat(*pose_r1)

    t = 0.0

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE: break

        if gui.is_pressed('d'): t = min(1.0, t + 0.01)
        if gui.is_pressed('a'): t = max(0.0, t - 0.01)

        # 姿态 A（暗紫，固定不动）
        compute_transform(ti.Vector([float(v) for v in pose_r0]))
        draw_cube(gui, 0x442266)

        # 姿态 B（暗蓝，固定不动）
        compute_transform(ti.Vector([float(v) for v in pose_r1]))
        draw_cube(gui, 0x113366)

        # ── 核心：SLERP 插值 → 转回欧拉角 → 原有 kernel ──
        q_t = slerp(qa, qb, t)
        curr_angles = quat_to_angles(q_t)   # 替换原来的线性插值
        compute_transform(curr_angles)
        draw_cube(gui, 0x33CCFF)            # 插值结果，亮蓝

        gui.show()

if __name__ == '__main__':
    main()