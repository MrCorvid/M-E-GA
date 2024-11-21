import json
import sys
from collections import deque
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
from moviepy.editor import ImageSequenceClip
import os

pygame.init()
WIDTH, HEIGHT = 1280, 720
pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Particle Simulation 3D Player")

WHITE = (1, 1, 1, 1)
BLACK = (0, 0, 0, 1)
YELLOW = (1, 1, 0, 1)
BLUE = (0, 0, 1, 1)
GRAY = (0.5, 0.5, 0.5, 1)
GREEN = (0, 1, 0, 1)

def init_gl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POINT_SMOOTH)
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

def draw_points(points, color, size):
    glPointSize(size)
    glBegin(GL_POINTS)
    glColor4f(*color)
    for point in points:
        glVertex3f(*[round(p) for p in point])
    glEnd()

def draw_path(path, color):
    glLineWidth(2)
    glBegin(GL_LINE_STRIP)
    glColor4f(*color)
    for point in path:
        glVertex3f(*[round(p) for p in point])
    glEnd()

class LogPlayer:
    def __init__(self, log_file, volume_size):
        self.log_file = log_file
        self.log_stream = open(log_file, 'r')
        self.current_frame = None
        self.frame_queue = deque(maxlen=10)
        self.rotation_x = 30
        self.rotation_y = 45
        self.zoom = -15
        self.paused = False
        self.playback_speed = 1
        self.frame_index = 0
        self.total_frames = sum(1 for _ in open(log_file))
        self.log_stream.seek(0)
        self.show_path = True
        self.volume_size = volume_size
        self.ff_rw_speed = 10
        self.rendering_video = False
        self.frame_images = []
        self.frame_navigation_mode = False
        self.end_of_frames = False
        self.use_shading = True

    def load_next_frame(self):
        if self.paused and not self.frame_navigation_mode:
            return

        frames_to_load = 1 if self.frame_navigation_mode else (
            self.ff_rw_speed if pygame.key.get_pressed()[pygame.K_RIGHT] else 1)
        for _ in range(frames_to_load):
            if not self.frame_queue:
                for _ in range(10):
                    line = self.log_stream.readline().strip()
                    if line:
                        frame = json.loads(line)
                        self.frame_queue.append(frame)
                        self.frame_index += 1
                    else:
                        self.end_of_frames = True
                        break

            if self.frame_queue:
                self.current_frame = self.frame_queue.popleft()
            else:
                if self.end_of_frames:
                    self.current_frame = None
                break

        if self.frame_navigation_mode:
            self.paused = True

    def load_previous_frame(self):
        frames_to_rewind = 1 if self.frame_navigation_mode else self.ff_rw_speed
        self.frame_index = max(0, self.frame_index - frames_to_rewind)
        self.log_stream.seek(0)
        self.frame_queue.clear()
        for _ in range(self.frame_index):
            self.log_stream.readline()
        self.load_next_frame()

        if self.frame_navigation_mode:
            self.paused = True

    def draw_wireframe(self):
        glColor4f(*GRAY)
        glLineWidth(1)
        glBegin(GL_LINES)

        # Draw the cube
        for i in range(2):
            for j in range(2):
                glVertex3f(0, i * self.volume_size, j * self.volume_size)
                glVertex3f(self.volume_size, i * self.volume_size, j * self.volume_size)

                glVertex3f(i * self.volume_size, 0, j * self.volume_size)
                glVertex3f(i * self.volume_size, self.volume_size, j * self.volume_size)

                glVertex3f(i * self.volume_size, j * self.volume_size, 0)
                glVertex3f(i * self.volume_size, j * self.volume_size, self.volume_size)

        glEnd()

    def draw_frame(self):
        if not self.current_frame:
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(45, (WIDTH / HEIGHT), 0.1, 100.0)

        glTranslatef(0.0, 0.0, self.zoom)

        # Calculate the center of the volume
        center = [self.volume_size / 2, self.volume_size / 2, self.volume_size / 2]

        # Translate to center, rotate, then translate back
        glTranslatef(*center)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glTranslatef(*[-c for c in center])

        self.draw_wireframe()

        # Set up lighting if shading is enabled
        if self.use_shading:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
            glLightfv(GL_LIGHT0, GL_DIFFUSE, WHITE)
        else:
            glDisable(GL_LIGHTING)

        # Draw particles
        particles = self.current_frame['particle_positions']
        draw_points(particles, WHITE if self.use_shading else BLUE, 5)

        if self.show_path:
            path = [(self.volume_size // 2, self.volume_size // 2, self.volume_size // 2)]  # Start at center
            x, y, z = self.volume_size // 2, self.volume_size // 2, self.volume_size // 2
            for move in self.current_frame['path']:
                if move in ['R', 'L', 'U', 'D', 'F', 'B']:
                    dx, dy, dz = {'R': (1, 0, 0), 'L': (-1, 0, 0), 'U': (0, 1, 0),
                                  'D': (0, -1, 0), 'F': (0, 0, 1), 'B': (0, 0, -1)}[move]
                    x = (x + dx) % self.volume_size
                    y = (y + dy) % self.volume_size
                    z = (z + dz) % self.volume_size
                    path.append((x, y, z))
            draw_path(path, GREEN)

        # Draw final position (yellow dot)
        glPointSize(8)
        glBegin(GL_POINTS)
        glColor4f(*YELLOW)
        final_pos = self.current_frame['final_position']
        glVertex3f(*[round(p) for p in final_pos])
        glEnd()

        self.draw_hud()

        pygame.display.flip()

        if self.rendering_video:
            self.save_frame_image()

    def save_frame_image(self):
        buffer = glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        image = np.flipud(image)
        self.frame_images.append(image)

    def create_video(self):
        if not self.frame_images:
            print("No frames captured. Cannot create video.")
            return

        output_path = "output_video.mp4"
        clip = ImageSequenceClip(self.frame_images, fps=30)
        clip.write_videofile(output_path)
        print(f"Video saved to {output_path}")
        self.frame_images.clear()

    def draw_text(self, text, x, y):
        font = pygame.font.Font(None, 24)
        text_surface = font.render(text, True, (0, 255, 0))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glWindowPos2d(x, HEIGHT - y)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    def draw_hud(self):
        text_lines = [
            f"Frame: {self.frame_index}/{self.total_frames}",
            f"Particles: {len(self.current_frame['particle_positions'])}",
            f"Fitness Score: {self.current_frame['fitness_score']:.2f}",
            f"Grouping Score: {self.current_frame['grouping_score']:.2f}",
            f"{'PAUSED' if self.paused else 'PLAYING'} (Space to toggle)",
            f"Playback speed: {self.playback_speed}x (Up/Down to adjust)",
            f"Path: {'Visible' if self.show_path else 'Hidden'} (P to toggle)",
            "Fast Forward: Right Arrow, Rewind: Left Arrow",
            f"Video Rendering: {'ON' if self.rendering_video else 'OFF'} (V to toggle)",
            f"Frame Navigation: {'ON' if self.frame_navigation_mode else 'OFF'} (N to toggle)",
            f"Shading: {'ON' if self.use_shading else 'OFF'} (S to toggle)"
        ]

        y_offset = 10
        for line in text_lines:
            self.draw_text(line, 10, y_offset)
            y_offset += 20

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    if event.buttons[0]:
                        self.rotation_y += event.rel[0]
                        self.rotation_x += event.rel[1]
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:
                        self.zoom += 0.5
                    elif event.button == 5:
                        self.zoom -= 0.5
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_UP:
                        self.playback_speed = min(4, self.playback_speed * 2)
                    elif event.key == pygame.K_DOWN:
                        self.playback_speed = max(0.25, self.playback_speed / 2)
                    elif event.key == pygame.K_p:
                        self.show_path = not self.show_path
                    elif event.key == pygame.K_v:
                        self.rendering_video = not self.rendering_video
                        if not self.rendering_video:
                            self.create_video()
                    elif event.key == pygame.K_n:
                        self.frame_navigation_mode = not self.frame_navigation_mode
                        if not self.frame_navigation_mode:
                            self.paused = False
                    elif event.key == pygame.K_s:
                        self.use_shading = not self.use_shading

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.load_previous_frame()
            elif keys[pygame.K_RIGHT]:
                self.load_next_frame()
            elif not self.paused:
                for _ in range(int(self.playback_speed)):
                    self.load_next_frame()

            if self.current_frame:
                self.draw_frame()
            elif self.end_of_frames:
                if self.rendering_video:
                    self.create_video()
                    self.rendering_video = False
                    print("Video rendering completed. Press any key to exit.")
                    pygame.event.wait()
                running = False
            else:
                running = False

            clock.tick(30)

        pygame.quit()
        self.log_stream.close()

def get_log_file():
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="Select log file",
                                               filetypes=(("JSONL files", "*.jsonl"), ("All files", "*.*")))
        return file_path

def get_volume_size():
    if len(sys.argv) > 2:
        return int(sys.argv[2])
    else:
        return int(input("Enter the volume size: "))

if __name__ == "__main__":
    pygame.init()
    display = (WIDTH, HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    init_gl()

    log_file = get_log_file()
    volume_size = get_volume_size()
    if log_file:
        player = LogPlayer(log_file, volume_size)
        player.run()
    else:
        print("No log file selected. Exiting.")