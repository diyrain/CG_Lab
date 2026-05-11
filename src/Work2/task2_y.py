import taichi as ti
import math

# 初始化 Taichi
ti.init(arch=ti.cpu)

# 常量定义
NUM_VERTICES = 8
NUM_EDGES = 12

# 声明 Field
vertices = ti.Vector.field(3, dtype=ti.f32, shape=NUM_VERTICES)
edges = ti.Vector.field(2, dtype=ti.i32, shape=NUM_EDGES)  # 存储边的顶点索引对
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=NUM_VERTICES)

@ti.func
def get_model_matrix(angle: ti.f32):
    """绕 Y 轴旋转的模型矩阵"""
    radians = angle * ti.math.pi / 180.0
    c = ti.cos(radians)
    s = ti.sin(radians)
    return ti.Matrix([
        [ c,  0.0,  s,  0.0],
        [ 0.0, 1.0,  0.0, 0.0],
        [-s,  0.0,  c,  0.0],
        [ 0.0, 0.0,  0.0, 1.0]
    ], dt=ti.f32)

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ], dt=ti.f32)

@ti.func
def get_projection_matrix(fovy: ti.f32, aspect: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """标准透视投影矩阵"""
    tan_half_fovy = ti.tan(fovy * ti.math.pi / 360.0)
    res = ti.Matrix.zero(ti.f32, 4, 4)
    res[0, 0] = 1.0 / (aspect * tan_half_fovy)
    res[1, 1] = 1.0 / (tan_half_fovy)
    res[2, 2] = -(zFar + zNear) / (zFar - zNear)
    res[2, 3] = -(2.0 * zFar * zNear) / (zFar - zNear)
    res[3, 2] = -1.0 
    return res 

@ti.kernel
def compute_transform(angle: ti.f32): 
    eye_pos = ti.Vector([0.0, 0.0, 6.0]) # 相机拉远一点看全景
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    # MVP 矩阵组合
    mvp = proj @ view @ model
    
    for i in range(NUM_VERTICES):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法
        w = v_clip[3]
        v_ndc = ti.Vector([v_clip[0] / w, v_clip[1] / w])
        
        # 映射到屏幕坐标 [0, 1]
        screen_coords[i] = (v_ndc + 1.0) / 2.0

def init_cube():
    # 1. 定义 8 个顶点 (中心在原点，边长为 2)
    v_list = [
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
    ]
    for i in range(8):
        vertices[i] = v_list[i]
    
    # 2. 定义 12 条边 (存储顶点的索引)
    e_list = [
        [0, 1], [1, 2], [2, 3], [3, 0], # 底面四条边
        [4, 5], [5, 6], [6, 7], [7, 4], # 顶面四条边
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接上下面的垂直边
    ]
    for i in range(12):
        edges[i] = e_list[i]

def main():
    init_cube()
    gui = ti.GUI("3D Cube Wireframe", res=(700, 700))
    angle = 0.0
    
    while gui.running:
        # 事件处理
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle -= 5.0
            elif gui.event.key == 'd':
                angle += 5.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        # 计算变换
        compute_transform(angle)
        
        # 渲染正方体 12 条边
        # 将 screen_coords 转换为 numpy 以便在循环中快速访问
        coords = screen_coords.to_numpy()
        edge_indices = edges.to_numpy()
        
        for i in range(NUM_EDGES):
            idx1, idx2 = edge_indices[i]
            # 绘制边，起始点 A 到 终止点 B
            gui.line(coords[idx1], coords[idx2], radius=2, color=0x00FF00)
        
        gui.show()

if __name__ == '__main__':
    main()