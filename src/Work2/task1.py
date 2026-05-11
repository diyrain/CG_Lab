import taichi as ti
import math

# 初始化 Taichi
ti.init(arch=ti.cpu)

# 声明 Field
vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=3)

@ti.func
def get_model_matrix(angle: ti.f32):
    radians = angle * ti.math.pi / 180.0
    c = ti.cos(radians)
    s = ti.sin(radians)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
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
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    
    # MVP 矩阵
    mvp = proj @ view @ model
    
    for i in range(3):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        w = v_clip[3]
        v_ndc = ti.Vector([v_clip[0] / w, v_clip[1] / w])
        
        screen_coords[i] = (v_ndc + 1.0) / 2.0

def main():
    vertices[0] = [1.0, -0.5, -2.0]
    vertices[1] = [0.0, 1.0, -2.0]
    vertices[2] = [-1.0, -0.5, -2.0]
    
    gui = ti.GUI("3D Transformation (Taichi)", res=(700, 700))
    angle = 0.0
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle += 10.0
            elif gui.event.key == 'd':
                angle -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        compute_transform(angle)
        
        a = screen_coords[0]
        b = screen_coords[1]
        c = screen_coords[2]
        
        gui.line(a, b, radius=2, color=0xFF0000)
        gui.line(b, c, radius=2, color=0x00FF00)
        gui.line(c, a, radius=2, color=0x0000FF)
        
        gui.show()

if __name__ == '__main__':
    main()