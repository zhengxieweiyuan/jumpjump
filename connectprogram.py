import time
import random
from PIL import Image, ImageGrab
import win32gui
import win32api
import win32con
import ctypes  # Required for DPI awareness
from jump_jump_helper import get_jump_distance
import os
# --- Configuration ---
# This coefficient determines the press duration for a given distance.
# You will likely need to adjust this value based on your computer's performance and screen resolution.
# - If the jump is too short, increase the value.
# - If the jump is too long, decrease the value.
# --- 可配置参数 ---
# 游戏窗口标题
GAME_WINDOW_TITLE = "跳一跳" 
# 模型文件路径
MODEL_PATH = "jump_jump_model.pth"
# 按压时间系数，需要根据实际情况微调
# 系数越大，按压时间越长
PRESS_COEFFICIENT = 2.6  #这个根据不同的电脑分辨率来
# 每次跳跃之间的间隔（秒）
JUMP_INTERVAL = 0.01
# 截图保存路径
SCREENSHOT_PATH = "screenshot.png"
# 结果图保存路径
RESULT_IMAGE_PATH = "result.png"
# 是否开启可视化（在结果图上绘制检测框和中心点）
VISUALIZE = True



# --- Main Program ---

def get_game_window_rect(window_title):
    """
    Finds the game window's client area and returns its absolute screen coordinates.
    This is more accurate as it excludes the window frame and handles DPI scaling.
    """
    try:
        # Make the script DPI aware to get correct coordinates on scaled displays
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        print("Warning: Could not set DPI awareness (might be an older OS).")

    try:
        hwnd = win32gui.FindWindow(None, window_title)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.2)  # Allow window to come to the foreground

            # Get the client rectangle (the area inside the window borders)
            client_rect = win32gui.GetClientRect(hwnd)
            
            # Convert the client rect's top-left and bottom-right corners to absolute screen coordinates
            left, top = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))
            right, bottom = win32gui.ClientToScreen(hwnd, (client_rect[2], client_rect[3]))
            
            game_rect = (left, top, right, bottom)
            
            # Sanity check: if the rect has zero width or height, something is wrong.
            if game_rect[2] - game_rect[0] == 0 or game_rect[3] - game_rect[1] == 0:
                 print(f"Error: Found window '{window_title}', but its size is zero. Is it minimized?")
                 return None

            return game_rect
        else:
            print(f"Error: Could not find window with title '{window_title}'.")
            print("Please make sure the 'Jump Jump' game is open, visible, and the title is correct.")
            return None
    except Exception as e:
        print(f"An error occurred while trying to find the game window: {e}")
        return None

def capture_screen(rect, filename="autojump_screenshot.png"):
    """
    Captures a screenshot of the specified area and saves it to a file.
    """
    try:
        img = ImageGrab.grab(bbox=rect)
        img.save(filename)
        return filename
    except Exception as e:
        print(f"An error occurred during screen capture: {e}")
        return None

def simulate_jump(press_time, rect):
    """
    Simulates a mouse press-and-hold (jump) within the game window.
    """
    # Generate random coordinates within the window to make clicks less predictable
    click_x = random.randint(rect[0] + 50, rect[2] - 50)
    click_y = random.randint(rect[1] + 150, rect[3] - 150)

    win32api.SetCursorPos((click_x, click_y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    # Convert press_time from milliseconds to seconds for time.sleep()
    time.sleep(press_time / 1000.0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def main():
    """
    主函数，循环执行截图、分析、跳跃的流程
    """
    # 确保DPI感知，防止截图错位
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        print("DPI awareness could not be set (not available on this version of Windows).")
    # 1. 找到游戏窗口并截图
    rect = get_game_window_rect(GAME_WINDOW_TITLE)
    if not rect:
        print(f"错误：未找到标题为 '{GAME_WINDOW_TITLE}' 的窗口，请确保游戏正在运行。") 
        return
    print(f"成功找到窗口，坐标: {rect}")
    n=1
    try:
        while True:
            try: 
                # 等待动画稳定
                time.sleep(1.3)
                capture_screen(rect, SCREENSHOT_PATH) 
                # 2. 使用模型计算跳跃距离 
                distance, result_img = get_jump_distance(MODEL_PATH, SCREENSHOT_PATH, visualize=VISUALIZE)
                 # 保存可视化结果图
                result_img.save(RESULT_IMAGE_PATH)
                 
                # 3. 如果成功，则执行跳跃
                if distance > 0: 
                    print(f"第{n}次成功检测到目标方块,跳跃距离:",f"{distance:.2f}")
                    press_time = distance * PRESS_COEFFICIENT 
                    # 4. 模拟鼠标点击 (修正参数顺序)
                    simulate_jump(press_time, rect) 
                    # 等待JUMP_INTERVAL秒后下一次跳跃 
                    time.sleep(JUMP_INTERVAL) 
                    n=n+1
                else:
                    break 
            except Exception as e:
                print(f"发生意外错误: {e}") 
                return
    finally:
        # This block can be left empty or used for other cleanup if needed.
        pass

if __name__ == '__main__':         
    main()
