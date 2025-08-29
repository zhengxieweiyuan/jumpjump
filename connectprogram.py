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
JUMP_INTERVAL = 1.5
# 截图保存路径
SCREENSHOT_PATH = "autojump_screenshot.png"
# 结果图保存路径
RESULT_IMAGE_PATH = "autojump_result.png"
# 是否开启可视化（在结果图上绘制检测框和中心点）
VISUALIZE = True

FAILED_IMAGE_SAVE_DIR =".\data\images"

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
    
    previous_screenshot_path = "previous_autojump_screenshot.png"
    try:
        while True:
            try: 
                # 等待动画稳定
                time.sleep(1.5)
                capture_screen(rect, SCREENSHOT_PATH)
                print(f"截图已保存至: {SCREENSHOT_PATH}")

                # 2. 使用模型计算跳跃距离
                print("正在分析图像，计算跳跃距离...")
                distance, result_img = get_jump_distance(MODEL_PATH, SCREENSHOT_PATH, visualize=VISUALIZE)

                # 3. 如果成功，则执行跳跃
                if distance is not None:
                    # 保存这张成功的截图，以便在下次失败时使用
                    with Image.open(SCREENSHOT_PATH) as img:
                        img.save(previous_screenshot_path)
                         
                    # 保存可视化结果图
                    result_img.save(RESULT_IMAGE_PATH)
                    print(f"分析结果图已保存至: {RESULT_IMAGE_PATH}")

                    press_time = distance * PRESS_COEFFICIENT
                    print(f"计算出的按压时间: {press_time:.2f} 毫秒")
                    
                    # 4. 模拟鼠标点击 (修正参数顺序)
                    simulate_jump(press_time, rect)
                    
                    # 等待下一次跳跃
                    print(f"等待 {JUMP_INTERVAL} 秒后进行下一次跳跃...")
                    time.sleep(JUMP_INTERVAL)
                else:
                    print("未能计算出跳跃距离，这很可能是因为上一次跳跃失败了。")
                    if os.path.exists(previous_screenshot_path):
                        if not os.path.exists(FAILED_IMAGE_SAVE_DIR):
                            os.makedirs(FAILED_IMAGE_SAVE_DIR, exist_ok=True)
                        
                        # 使用时间戳生成唯一文件名
                        timestamp_suffix = int(time.time())
                        fail_filename = f"fail_picture4_{timestamp_suffix}.png"
                        fail_save_path = os.path.join(FAILED_IMAGE_SAVE_DIR, fail_filename)
                        
                        # 复制导致失败的上一张截图到失败目录
                        with Image.open(previous_screenshot_path) as img:
                            img.save(fail_save_path)
                        print(f"导致失败的截图已保存至: {fail_save_path}")
                    else:
                        print("未找到上一张截图，无法保存导致失败的图片。")
                    return # 退出程序
            except Exception as e:
                print(f"发生意外错误: {e}") 
                return
    finally:
        # 清理临时的截图文件
        if os.path.exists(previous_screenshot_path):
            os.remove(previous_screenshot_path)
if __name__ == '__main__':

    if not os.path.exists(FAILED_IMAGE_SAVE_DIR):
                  os.makedirs(FAILED_IMAGE_SAVE_DIR, exist_ok=True)
                
    main()
