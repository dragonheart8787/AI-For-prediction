# WebBit AGI預測系統控制器
# 使用WebBit開發板控制和顯示AGI預測結果
# 
# 硬體需求:
# - WebBit開發板
# - OLED 顯示器 (SSD1306)
# - RGB LED燈條
# - 蜂鳴器
# - 按鈕 x5 (金融/天氣/醫療/能源/語言)
# - 光敏電阻

from machine import Pin, I2C, PWM, ADC
import network
import urequests
import ujson
import time
import uasyncio
from ssd1306 import SSD1306_I2C

class WebBitAGIController:
    def __init__(self):
        # 初始化硬體
        self.setup_hardware()
        
        # WiFi設定
        self.wifi_ssid = "YOUR_WIFI_SSID"
        self.wifi_password = "YOUR_WIFI_PASSWORD"
        
        # AGI系統API地址
        self.agi_api_url = "http://192.168.1.100:8000"  # 修改為您的AGI系統IP
        
        # 預測狀態
        self.current_prediction = None
        self.last_confidence = 0
        self.is_predicting = False
        
        # 顯示狀態
        self.display_brightness = 100
        
        print("🤖 WebBit AGI控制器已初始化")
    
    def setup_hardware(self):
        """初始化硬體設備"""
        try:
            # I2C設定 (OLED顯示器)
            self.i2c = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
            self.oled = SSD1306_I2C(128, 64, self.i2c)
            
            # 按鈕設定 (5個預測功能按鈕)
            self.btn_financial = Pin(15, Pin.IN, Pin.PULL_UP)
            self.btn_weather = Pin(2, Pin.IN, Pin.PULL_UP)
            self.btn_medical = Pin(4, Pin.IN, Pin.PULL_UP)
            self.btn_energy = Pin(16, Pin.IN, Pin.PULL_UP)
            self.btn_language = Pin(17, Pin.IN, Pin.PULL_UP)
            
            # RGB LED設定 (置信度顯示)
            self.led_red = PWM(Pin(25), freq=1000)
            self.led_green = PWM(Pin(26), freq=1000)
            self.led_blue = PWM(Pin(27), freq=1000)
            
            # 蜂鳴器設定
            self.buzzer = PWM(Pin(18), freq=2000)
            
            # 光敏電阻設定 (亮度調節)
            self.light_sensor = ADC(Pin(36))
            self.light_sensor.atten(ADC.ATTN_11DB)
            
            # 狀態LED (系統狀態指示)
            self.status_led = Pin(5, Pin.OUT)
            
            print("✅ 硬體初始化完成")
            
        except Exception as e:
            print(f"❌ 硬體初始化失敗: {e}")
    
    def connect_wifi(self):
        """連接WiFi"""
        wlan = network.WLAN(network.STA_IF)
        wlan.active(True)
        
        if not wlan.isconnected():
            print("🔄 正在連接WiFi...")
            wlan.connect(self.wifi_ssid, self.wifi_password)
            
            # 等待連接
            timeout = 10
            while not wlan.isconnected() and timeout > 0:
                time.sleep(1)
                timeout -= 1
                self.status_led.value(timeout % 2)  # 閃爍指示
            
            if wlan.isconnected():
                print(f"✅ WiFi已連接: {wlan.ifconfig()[0]}")
                self.status_led.value(1)  # 常亮表示連接成功
                self.display_message("WiFi Connected", wlan.ifconfig()[0])
                return True
            else:
                print("❌ WiFi連接失敗")
                self.display_message("WiFi Failed", "Check Settings")
                return False
        else:
            print("✅ WiFi已連接")
            return True
    
    def display_message(self, title, content="", line3="", line4=""):
        """在OLED上顯示訊息"""
        try:
            self.oled.fill(0)
            self.oled.text(title[:16], 0, 0)
            if content:
                self.oled.text(content[:16], 0, 16)
            if line3:
                self.oled.text(line3[:16], 0, 32)
            if line4:
                self.oled.text(line4[:16], 0, 48)
            self.oled.show()
        except Exception as e:
            print(f"顯示錯誤: {e}")
    
    def set_confidence_led(self, confidence):
        """根據置信度設定LED顏色"""
        try:
            if confidence >= 0.8:
                # 高置信度 - 綠燈
                self.led_red.duty(0)
                self.led_green.duty(1023)
                self.led_blue.duty(0)
            elif confidence >= 0.6:
                # 中等置信度 - 黃燈
                self.led_red.duty(1023)
                self.led_green.duty(1023)
                self.led_blue.duty(0)
            else:
                # 低置信度 - 紅燈
                self.led_red.duty(1023)
                self.led_green.duty(0)
                self.led_blue.duty(0)
        except Exception as e:
            print(f"LED設定錯誤: {e}")
    
    def play_notification(self, success=True):
        """播放提示音"""
        try:
            if success:
                # 成功音效 - 上升音調
                for freq in [440, 554, 659]:
                    self.buzzer.freq(freq)
                    self.buzzer.duty(512)
                    time.sleep(0.2)
                    self.buzzer.duty(0)
                    time.sleep(0.1)
            else:
                # 錯誤音效 - 下降音調
                for freq in [659, 554, 440]:
                    self.buzzer.freq(freq)
                    self.buzzer.duty(512)
                    time.sleep(0.3)
                    self.buzzer.duty(0)
                    time.sleep(0.1)
        except Exception as e:
            print(f"音效播放錯誤: {e}")
    
    def read_light_level(self):
        """讀取環境亮度並調整顯示"""
        try:
            light_value = self.light_sensor.read()
            # 將ADC值 (0-4095) 映射到亮度 (20-100)
            brightness = int((light_value / 4095) * 80 + 20)
            self.display_brightness = brightness
            return light_value
        except Exception as e:
            print(f"光度感測錯誤: {e}")
            return 2000  # 默認值
    
    async def make_prediction_request(self, domain, task_type, data):
        """向AGI系統發送預測請求"""
        try:
            url = f"{self.agi_api_url}/predict/{domain}"
            
            payload = {
                "task_type": task_type,
                "data": data
            }
            
            print(f"🔄 發送 {domain} 預測請求...")
            self.display_message("Predicting...", domain.upper(), "Please wait")
            
            # 發送HTTP請求
            response = urequests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            
            if response.status_code == 200:
                result = response.json()
                response.close()
                return result
            else:
                print(f"❌ API請求失敗: {response.status_code}")
                response.close()
                return None
                
        except Exception as e:
            print(f"❌ 預測請求錯誤: {e}")
            return None
    
    async def handle_financial_prediction(self):
        """處理金融預測"""
        self.is_predicting = True
        self.display_message("Financial", "Prediction", "Loading...")
        
        # 模擬歷史數據
        historical_data = [100 + i + (i % 5) * 2 for i in range(30)]
        
        data = {
            "asset_type": "stocks",
            "timeframe": "1d",
            "historical_data": historical_data
        }
        
        result = await self.make_prediction_request("financial", "short_term_forecast", data)
        
        if result:
            confidence = result.get('confidence', 0)
            predictions = result.get('predictions', {})
            next_price = predictions.get('next_price', 0)
            
            self.set_confidence_led(confidence)
            self.display_message("Financial", f"Price: ${next_price:.2f}", f"Conf: {confidence:.1%}", "SUCCESS")
            self.play_notification(True)
            
            print(f"💰 金融預測完成 - 價格: ${next_price:.2f}, 置信度: {confidence:.1%}")
        else:
            self.set_confidence_led(0)
            self.display_message("Financial", "FAILED", "Check network", "Try again")
            self.play_notification(False)
        
        self.is_predicting = False
    
    async def handle_weather_prediction(self):
        """處理天氣預測"""
        self.is_predicting = True
        self.display_message("Weather", "Prediction", "Loading...")
        
        data = {
            "location": {"lat": 25.0330, "lon": 121.5654},
            "forecast_hours": 24
        }
        
        result = await self.make_prediction_request("weather", "weather_forecast", data)
        
        if result:
            confidence = result.get('confidence', 0)
            predictions = result.get('predictions', {})
            current_conditions = predictions.get('current_conditions', {})
            temperature = current_conditions.get('temperature', 0)
            
            self.set_confidence_led(confidence)
            self.display_message("Weather", f"Temp: {temperature:.1f}C", f"Conf: {confidence:.1%}", "SUCCESS")
            self.play_notification(True)
            
            print(f"🌤️ 天氣預測完成 - 溫度: {temperature:.1f}°C, 置信度: {confidence:.1%}")
        else:
            self.set_confidence_led(0)
            self.display_message("Weather", "FAILED", "Check network", "Try again")
            self.play_notification(False)
        
        self.is_predicting = False
    
    async def handle_medical_prediction(self):
        """處理醫療預測"""
        self.is_predicting = True
        self.display_message("Medical", "Prediction", "Loading...")
        
        data = {
            "patient_data": {"age": 65, "gender": "male"},
            "medical_history": ["diabetes", "hypertension"]
        }
        
        result = await self.make_prediction_request("medical", "readmission_risk", data)
        
        if result:
            confidence = result.get('confidence', 0)
            predictions = result.get('predictions', {})
            risk_level = predictions.get('risk_level', 'unknown')
            risk_prob = predictions.get('readmission_probability', 0)
            
            self.set_confidence_led(confidence)
            self.display_message("Medical", f"Risk: {risk_level}", f"Prob: {risk_prob:.1%}", f"Conf: {confidence:.1%}")
            self.play_notification(True)
            
            print(f"⚕️ 醫療預測完成 - 風險: {risk_level}, 置信度: {confidence:.1%}")
        else:
            self.set_confidence_led(0)
            self.display_message("Medical", "FAILED", "Check network", "Try again")
            self.play_notification(False)
        
        self.is_predicting = False
    
    async def handle_energy_prediction(self):
        """處理能源預測"""
        self.is_predicting = True
        self.display_message("Energy", "Prediction", "Loading...")
        
        # 模擬歷史負載數據
        historical_data = [25000 + i * 100 + (i % 24) * 500 for i in range(168)]
        
        data = {
            "energy_type": "electricity",
            "region": "taiwan",
            "historical_data": historical_data,
            "forecast_hours": 24
        }
        
        result = await self.make_prediction_request("energy", "load_forecast", data)
        
        if result:
            confidence = result.get('confidence', 0)
            predictions = result.get('predictions', {})
            summary = predictions.get('summary', {})
            peak_load = summary.get('peak_load_mw', 0)
            
            self.set_confidence_led(confidence)
            self.display_message("Energy", f"Peak: {peak_load:.0f}MW", f"Conf: {confidence:.1%}", "SUCCESS")
            self.play_notification(True)
            
            print(f"⚡ 能源預測完成 - 峰值: {peak_load:.0f}MW, 置信度: {confidence:.1%}")
        else:
            self.set_confidence_led(0)
            self.display_message("Energy", "FAILED", "Check network", "Try again")
            self.play_notification(False)
        
        self.is_predicting = False
    
    async def handle_language_prediction(self):
        """處理語言預測"""
        self.is_predicting = True
        self.display_message("Language", "Prediction", "Loading...")
        
        data = {
            "text": "人工智慧的未來發展",
            "language": "zh-TW",
            "max_length": 100
        }
        
        result = await self.make_prediction_request("language", "text_generation", data)
        
        if result:
            confidence = result.get('confidence', 0)
            predictions = result.get('predictions', {})
            generated_text = predictions.get('generated_text', '')
            word_count = predictions.get('word_count', 0)
            
            self.set_confidence_led(confidence)
            self.display_message("Language", f"Words: {word_count}", f"Conf: {confidence:.1%}", "SUCCESS")
            self.play_notification(True)
            
            print(f"💬 語言預測完成 - 字數: {word_count}, 置信度: {confidence:.1%}")
        else:
            self.set_confidence_led(0)
            self.display_message("Language", "FAILED", "Check network", "Try again")
            self.play_notification(False)
        
        self.is_predicting = False
    
    async def monitor_buttons(self):
        """監控按鈕輸入"""
        button_handlers = {
            self.btn_financial: ("Financial", self.handle_financial_prediction),
            self.btn_weather: ("Weather", self.handle_weather_prediction),
            self.btn_medical: ("Medical", self.handle_medical_prediction),
            self.btn_energy: ("Energy", self.handle_energy_prediction),
            self.btn_language: ("Language", self.handle_language_prediction)
        }
        
        button_states = {btn: False for btn in button_handlers.keys()}
        
        while True:
            # 檢查每個按鈕
            for button, (name, handler) in button_handlers.items():
                current_state = not button.value()  # 按鈕按下時為低電位
                
                # 檢測按鈕按下事件 (從未按下到按下)
                if current_state and not button_states[button] and not self.is_predicting:
                    print(f"🔘 {name} 按鈕被按下")
                    button_states[button] = True
                    
                    # 非同步執行預測
                    uasyncio.create_task(handler())
                
                elif not current_state:
                    button_states[button] = False
            
            await uasyncio.sleep_ms(50)  # 50ms 掃描間隔
    
    async def monitor_environment(self):
        """監控環境狀態"""
        while True:
            try:
                # 讀取光度並調整顯示亮度
                light_level = self.read_light_level()
                
                # 如果沒有在執行預測，顯示系統狀態
                if not self.is_predicting:
                    # 每10秒更新一次狀態顯示
                    self.display_message("AGI Controller", "Ready", f"Light: {light_level}", "Press button")
                
                await uasyncio.sleep(10)  # 10秒檢查一次
                
            except Exception as e:
                print(f"環境監控錯誤: {e}")
                await uasyncio.sleep(5)
    
    async def main_loop(self):
        """主程序循環"""
        print("🚀 WebBit AGI控制器啟動")
        
        # 初始化顯示
        self.display_message("AGI Controller", "Starting...", "Please wait")
        
        # 連接WiFi
        if not self.connect_wifi():
            self.display_message("ERROR", "WiFi Failed", "Check config", "Restart")
            return
        
        # 顯示就緒狀態
        self.display_message("AGI Controller", "Ready", "WiFi Connected", "Press button")
        self.set_confidence_led(0.5)  # 中性指示燈
        
        # 啟動監控任務
        button_task = uasyncio.create_task(self.monitor_buttons())
        env_task = uasyncio.create_task(self.monitor_environment())
        
        print("✅ 系統就緒，等待按鈕輸入...")
        
        # 等待任務完成
        await uasyncio.gather(button_task, env_task)

# 主程序
def main():
    """主程序入口"""
    try:
        controller = WebBitAGIController()
        uasyncio.run(controller.main_loop())
    except KeyboardInterrupt:
        print("\n👋 程序中斷")
    except Exception as e:
        print(f"❌ 程序錯誤: {e}")

if __name__ == "__main__":
    main() 