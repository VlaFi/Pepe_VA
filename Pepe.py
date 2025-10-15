import json
import threading
from queue import Queue
import time
import os
import re
import numpy as np
import random
import sqlite3
from datetime import datetime
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from llama_cpp import Llama
import torch
import sounddevice as sd_play
from scipy import signal
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
from threading import Thread
import requests
from bs4 import BeautifulSoup

VOSK_MODEL_PATH = "vosk-model-ru-0.42"
LLAMA_MODEL_PATH = "saiga_gemma3_12b.Q4_K_M.gguf"
TTS_SAMPLE_RATE = 48000
SAMPLE_RATE = 48000
CHANNELS = 1
DTYPE = np.int16
LLAMA_CTX_SIZE = 4096
LLAMA_N_GPU_LAYERS = -1  
MAX_DIALOG_HISTORY = 10
DB_PATH = "dialog_history.db"
MAX_CONTEXT_LENGTH = 3000

class VoiceAssistant:
    def __init__(self):
        self.is_recording = False
        self.audio_queue = Queue()
        self.dialog_history = []
        self.conn = None
        self.cursor = None
        self.user_name = None
        self.setup_database()
        self.setup_models()
        
    def setup_database(self):
        try:
            self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS dialogs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print("База данных инициализирована успешно")
        except Exception as e:
            print(f"Ошибка при инициализации базы данных: {e}")
        
    def setup_models(self):
        print("Инициализация моделей...")
        print("Загрузка модели для распознавания речи (Vosk)...")
        self.vosk_model = Model(VOSK_MODEL_PATH)
        self.recognizer = KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
        self.recognizer.SetWords(True)
        
        print("Загрузка языковой модели (Saiga Gemma3 12B)...")
        if torch.cuda.is_available():
            print(f"CUDA доступна, используем GPU: {torch.cuda.get_device_name(0)}")
            try:
                self.llm = Llama(
                    model_path=LLAMA_MODEL_PATH,
                    n_ctx=LLAMA_CTX_SIZE,
                    n_gpu_layers=LLAMA_N_GPU_LAYERS,
                    n_threads=6,
                    verbose=True
                )
                print("Модель загружена на GPU")
            except Exception as e:
                print(f"Ошибка при загрузке модели на GPU: {e}")
                print("Пробуем загрузить на CPU...")
                self.llm = Llama(
                    model_path=LLAMA_MODEL_PATH,
                    n_ctx=LLAMA_CTX_SIZE,
                    n_gpu_layers=0,
                    n_threads=8,
                    verbose=False
                )
        else:
            print("CUDA недоступна, используем CPU")
            self.llm = Llama(
                model_path=LLAMA_MODEL_PATH,
                n_ctx=LLAMA_CTX_SIZE,
                n_gpu_layers=0,
                n_threads=8,
                verbose=False
            )
        
        print("Загрузка модели для синтеза речи (Silero)...")
        if torch.cuda.is_available():
            self.tts_device = torch.device('cuda')
            print(f"Используемое устройство для TTS: {self.tts_device}")
        else:
            self.tts_device = torch.device('cpu')
            print(f"Используемое устройство для TTS: {self.tts_device}")
            
        torch.set_num_threads(4)
        
        self.tts_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language='ru',
            speaker='v3_1_ru'
        )
        self.tts_model.to(self.tts_device)
        
        print("Все модели загружены!")
        self.load_dialog_history()
    
    def get_user_profiles(self):
        """Получить список всех профилей пользователей"""
        try:
            self.cursor.execute("SELECT name FROM user_profiles ORDER BY created_at DESC")
            return [row[0] for row in self.cursor.fetchall()]
        except Exception as e:
            print(f"Ошибка при получении профилей: {e}")
            return []
    
    def save_user_profile(self, name):
        """Сохранить профиль пользователя"""
        try:
            self.cursor.execute(
                "INSERT OR IGNORE INTO user_profiles (name) VALUES (?)",
                (name,)
            )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Ошибка при сохранении профиля: {e}")
            return False
    
    def set_user_name(self, name):
        """Установить имя текущего пользователя"""
        if name:
            self.user_name = name
            # Сохраняем профиль
            self.save_user_profile(name)
            return True
        return False
    
    def search_wikipedia(self, query, lang='ru'):
        """Поиск в Википедии на русском языке"""
        try:
            url = f"https://{lang}.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'utf8': 1,
                'srlimit': 3
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['query']['search']:
                    # Получаем первую статью
                    page_id = data['query']['search'][0]['pageid']
                    
                    # Получаем содержание статьи
                    content_params = {
                        'action': 'query',
                        'format': 'json',
                        'pageids': page_id,
                        'prop': 'extracts',
                        'exintro': True,
                        'explaintext': True,
                        'utf8': 1
                    }
                    
                    content_response = requests.get(url, params=content_params, timeout=10)
                    if content_response.status_code == 200:
                        content_data = content_response.json()
                        page = content_data['query']['pages'][str(page_id)]
                        return page.get('extract', 'Информация не найдена')
            return None
        except Exception as e:
            print(f"Ошибка при поиске в Википедии: {e}")
            return None
    
    def load_dialog_history(self):
        try:
            if self.cursor is None:
                return
                
            self.cursor.execute("SELECT role, message, timestamp FROM dialogs ORDER BY timestamp DESC LIMIT ?", (MAX_DIALOG_HISTORY,))
            rows = self.cursor.fetchall()
            
            self.dialog_history = []
            for row in rows:
                self.dialog_history.insert(0, {
                    "role": row[0],
                    "message": row[1],
                    "timestamp": row[2]
                })
                
            print(f"Загружено {len(self.dialog_history)} записей истории диалогов")
        except Exception as e:
            print(f"Ошибка при загрузке истории диалогов: {e}")
    
    def add_to_history(self, role, message):
        clean_message = re.sub(r'<\/?s>|\[INST\]|\[\/INST\]', '', message)
        clean_message = clean_message.strip()
        
        if clean_message:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dialog_entry = {
                "role": role,
                "message": clean_message,
                "timestamp": timestamp
            }
            
            self.dialog_history.append(dialog_entry)
            
            if self.cursor is not None:
                try:
                    self.cursor.execute(
                        "INSERT INTO dialogs (role, message, timestamp) VALUES (?, ?, ?)",
                        (role, clean_message, timestamp)
                    )
                    self.conn.commit()
                except Exception as e:
                    print(f"Ошибка при сохранении в базу данных: {e}")
            
            if len(self.dialog_history) > MAX_DIALOG_HISTORY:
                self.dialog_history = self.dialog_history[-MAX_DIALOG_HISTORY:]
    
    def save_dialog_history(self):
        try:
            with open("dialog_history.json", "w", encoding="utf-8") as f:
                json.dump(self.dialog_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка при сохранении истории диалогов: {e}")
    
    def get_conversation_context(self):
        """Получение контекста диалога (только последние 3-4 реплики)"""
        if not self.dialog_history:
            return ""
        
        # Берем только последние несколько реплик для контекста
        recent_history = self.dialog_history[-4:] if len(self.dialog_history) > 4 else self.dialog_history
        
        context = "Предыдущий диалог:\n"
        for dialog in recent_history:
            role = "Пользователь" if dialog["role"] == "Пользователь" else "Пепе"
            context += f"{role}: {dialog['message']}\n"
        
        # Ограничиваем длину контекста
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[-MAX_CONTEXT_LENGTH:]
        
        return context

    def search_in_history(self, query, limit=5):
        try:
            if self.cursor is None:
                results = []
                for dialog in self.dialog_history:
                    if query.lower() in dialog["message"].lower():
                        results.append((dialog["role"], dialog["message"], dialog["timestamp"]))
                        if len(results) >= limit:
                            break
                return results
                
            self.cursor.execute(
                "SELECT role, message, timestamp FROM dialogs WHERE message LIKE ? ORDER BY timestamp DESC LIMIT ?",
                (f"%{query}%", limit)
            )
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Ошибка при поиске в истории: {e}")
            return []

    def apply_glados_effect(self, audio_array, sample_rate=48000):
        audio_array = audio_array * 2.0
        
        impulse_response = np.exp(-np.linspace(0, 5, int(sample_rate * 0.2)))
        audio_array = np.convolve(audio_array, impulse_response, mode='same')
        
        b, a = signal.butter(4, [800, 5000], btype='bandpass', fs=sample_rate)
        audio_array = signal.filtfilt(b, a, audio_array)
        
        audio_array = np.tanh(audio_array * 2.5)
        
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val * 0.9
            
        return audio_array
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
    
        volume_norm = np.linalg.norm(indata) / frames
        if volume_norm > 0.01:
            self.audio_queue.put(bytes(indata))

    def start_recording(self):
        if self.is_recording:
            return
            
        self.is_recording = True
        self.audio_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=8000,
            dtype=DTYPE,
            channels=CHANNELS,
            callback=self.audio_callback
        )
        self.audio_stream.start()
    
    def stop_recording(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
    
    def listen_and_recognize(self):
        if not self.is_recording:
            self.start_recording()
        
        print("Говорите...")
        audio_data = b''
        silence_counter = 0
        start_time = time.time()
    
        while time.time() - start_time < 7:
            try:
                data = self.audio_queue.get(timeout=0.1)
                audio_data += data
            
                audio_array = np.frombuffer(data, dtype=np.int16)
                volume = np.sqrt(np.mean(audio_array**2))
            
                if volume < 1000:
                    silence_counter += 1
                else:
                    silence_counter = 0
                
                if silence_counter > 10:
                    break
                
            except:
                pass
    
        self.stop_recording()
    
        if len(audio_data) < 1000:
            print("Аудио не содержит речи или слишком короткое")
            return None
        
        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.Result())
            recognized_text = result.get('text', '').strip()
        
            if recognized_text and len(recognized_text.split()) >= 1:
                print(f"Распознано: {recognized_text}")
                self.add_to_history("Пользователь", recognized_text)
                return recognized_text
            else:
                print("Распознан недостаточно осмысленный текст")
                return None
    
        return None
    
    def generate_answer(self, question, gui_callback=None):
        if gui_callback:
            gui_callback("🤔 Думаю...")
        
        print(f"Получен вопрос: {question}")
        
        # Обработка специальных запросов
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["время", "час", "сколько времени"]):
            current_time = datetime.now().strftime("%H:%M")
            response = f"Сейчас {current_time}"
            self.add_to_history("Пепе", response)
            print(f"Ответ: {response}")
            return response
        
        if any(word in question_lower for word in ["как тебя зовут", "твое имя", "представься"]):
            response = "Меня зовут Пепе. Я твой голосовой помощник."
            if self.user_name:
                response = f"Меня зовут Пепе. Приятно познакомиться, {self.user_name}!"
            self.add_to_history("Пепе", response)
            print(f"Ответ: {response}")
            return response
            
        if self.user_name and any(word in question_lower for word in ["как меня зовут", "мое имя", "забыла как меня"]):
            response = f"Тебя зовут {self.user_name}!"
            self.add_to_history("Пепе", response)
            print(f"Ответ: {response}")
            return response
        
        # Обработка команды для установки имени
        if question_lower.startswith("запомни моё имя") or question_lower.startswith("мое имя"):
            name = re.sub(r'запомни моё имя|мое имя', '', question, flags=re.IGNORECASE).strip()
            if name:
                if self.set_user_name(name):
                    response = f"Хорошо, я запомнила что тебя зовут {name}!"
                else:
                    response = "Не удалось сохранить имя. Попробуй еще раз."
                self.add_to_history("Пепе", response)
                return response
        
        # Проверяем, нужен ли поиск в Википедии (только для фактологических вопросов)
        needs_wiki_search = any(word in question_lower for word in ["что такое", "кто такой", "определение", "означает", "что значит", "расскажи про", "объясни"]) and not any(word in question_lower for word in ["ты", "тебе", "твои"])
        
        if needs_wiki_search:
            if gui_callback:
                gui_callback("🌐 Ищу информацию в Википедии...")
            
            # Ищем в Википедии
            wiki_result = self.search_wikipedia(question)
            if wiki_result:
                # Ограничиваем длину ответа
                if len(wiki_result) > 1000:
                    wiki_result = wiki_result[:1000] + "..."
                
                response = f"Согласно Википедии:\n\n{wiki_result}"
                self.add_to_history("Пепе", response)
                print(f"Ответ из Википедии: {response}")
                return response

        try:
            context = self.get_conversation_context()
            print(f"Контекст диалога: {context[:200]}...")  # Сокращаем вывод контекста
            
            current_time = datetime.now().strftime("%H:%M")
            system_prompt = f"""Ты — Пепе. Ты дружелюбная, эмпатичная девушка ИИ.
Текущее время: {current_time}.
Отвечай подробно, но естественно. Поддерживай беседу и задавай уточняющие вопросы.
Будь умной и информативной. Если не знаешь ответа, так и скажи.
Отвечай на русском языке.
"""
            
            if self.user_name:
                system_prompt += f"Пользователя зовут {self.user_name}. Обращайся к нему по имени.\n"

            prompt = f"{system_prompt}\n\n{context}\n\nПользователь: {question}\nПепе:"
            print(f"Полный промпт: {prompt[:300]}...")  # Сокращаем вывод промпта

            if gui_callback:
                gui_callback("🧠 Генерирую ответ...")

            output = self.llm(
                prompt,
                max_tokens=500,  # Увеличиваем для более подробных ответов
                stop=["Пользователь:", "Человек:", "Пепе:", "\n\n", "###", "<|"],
                echo=False,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.3  # Увеличиваем штраф за повторения
            )
        
            response = output['choices'][0]['text'].strip()
            print(f"Сырой ответ от модели: {response}")
        
            # Улучшенная очистка ответа
            stop_phrases = [
                "</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", 
                "Пользователь:", "Человек:", "Пеpe:", "###", "<|im_end|>",
                "```", "***", "---", "___", "&quot;", "&amp;"
            ]
        
            for phrase in stop_phrases:
                if phrase in response:
                    response = response.split(phrase)[0].strip()
            
            # Удаляем повторяющиеся приветствия
            response = re.sub(r'^.*?[Пп]ривет.*?\.\s*', '', response)
            
            if not response or len(response) < 5:
                response = "Извини, я не совсем поняла вопрос. Можешь переформулировать?"
            
            # Убедимся, что ответ не слишком длинный для синтеза речи
            if len(response) > 500:
                response = response[:497] + "..."
        
            self.add_to_history("Пеpe", response)
            print(f"Очищенный ответ: {response}")

            return response
        
        except Exception as e:
            print(f"Ошибка при генерации ответа: {e}")
            response = "Извини, у меня возникли проблемы с обработкой запроса. Попробуй задать вопрос по-другому."
            
            self.add_to_history("Пеpe", response)
            return response

    def speak_text(self, text):
        if not text:
            return
        
        try:
            clean_text = re.sub(r'^-+\s*|-+$', '', text)
            clean_text = clean_text.strip()
        
            if not clean_text:
                return
            
            print(f"Синтез речи для текста: {clean_text}")
            audio = self.tts_model.apply_tts(
                text=clean_text,
                speaker='xenia',
                sample_rate=TTS_SAMPLE_RATE,
                put_accent=True,
                put_yo=True
            )
            
            audio_numpy = audio.numpy()
            print(f"Аудио сгенерировано, форма: {audio_numpy.shape}, макс. значение: {np.max(np.abs(audio_numpy))}")
            
            audio_processed = self.apply_glados_effect(audio_numpy, TTS_SAMPLE_RATE)
            print(f"Аудио обработано, макс. значение: {np.max(np.abs(audio_processed))}")
        
            from scipy.io import wavfile
            wavfile.write("debug_audio.wav", 16000, (audio_processed * 32767).astype(np.int16))
            print("Аудио сохранено в debug_audio.wav для проверки")
            
            print("Воспроизведение аудио...")
            sd_play.play(audio_processed, samplerate=TTS_SAMPLE_RATE)
            sd_play.wait()
            print("Воспроизведение завершено")
            
        except Exception as e:
            print(f"Ошибка при синтезе речи: {e}")
    
    def process_text_input(self, text, gui_callback=None):
        if not text.strip():
            return None
            
        self.add_to_history("Пользователь", text)
        response = self.generate_answer(text, gui_callback)
        self.speak_text(response)
        
        return response

class AssistantGUI:
    def __init__(self, assistant):
        self.assistant = assistant
        self.root = tk.Tk()
        self.root.title("Пепе")
        self.root.geometry("1000x700")
        self.current_response = ""
        self.is_generating = False
        
        self.setup_dark_theme()
        self.setup_ui()
        self.show_profile_selection()

    def show_profile_selection(self):
        """Показать окно выбора профиля при запуске"""
        profile_window = tk.Toplevel(self.root)
        profile_window.title("Выбор профиля")
        profile_window.geometry("400x300")
        profile_window.configure(bg='#1a1a1a')
        profile_window.grab_set()  # Модальное окно
        profile_window.transient(self.root)  # Поверх главного окна
        
        # Центрируем окно
        profile_window.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (300 // 2)
        profile_window.geometry(f"400x300+{x}+{y}")
        
        ttk.Label(profile_window, text="Выберите или создайте профиль", 
                 font=("Courier", 12)).pack(pady=20)
        
        # Список существующих профилей
        profiles = self.assistant.get_user_profiles()
        
        if profiles:
            ttk.Label(profile_window, text="Существующие профили:").pack(pady=5)
            
            for profile in profiles:
                btn = ttk.Button(profile_window, text=profile, 
                               command=lambda p=profile: self.select_profile(p, profile_window))
                btn.pack(pady=2)
        
        # Поле для создания нового профиля
        ttk.Label(profile_window, text="Или создайте новый:").pack(pady=10)
        
        new_name_var = tk.StringVar()
        ttk.Entry(profile_window, textvariable=new_name_var, width=20).pack(pady=5)
        
        def create_new_profile():
            name = new_name_var.get().strip()
            if name:
                if self.assistant.set_user_name(name):
                    profile_window.destroy()
                    messagebox.showinfo("Успех", f"Профиль {name} создан и выбран!")
                else:
                    messagebox.showerror("Ошибка", "Не удалось создать профиль")
            else:
                messagebox.showwarning("Внимание", "Введите имя профиля")
        
        ttk.Button(profile_window, text="Создать профиль", 
                  command=create_new_profile).pack(pady=10)
    
    def select_profile(self, profile_name, window):
        """Выбрать существующий профиль"""
        self.assistant.user_name = profile_name
        window.destroy()
        messagebox.showinfo("Успех", f"Профиль {profile_name} выбран!")

    def setup_dark_theme(self):
        self.root.configure(bg='#1a1a1a')
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('.', background='#1a1a1a', foreground='#ff8800')
        self.style.configure('TFrame', background='#1a1a1a')
        self.style.configure('TLabel', background='#1a1a1a', foreground='#ff8800', font=('Courier', 10))
        self.style.configure('TButton', background='#333333', foreground='#ff8800', 
                           font=('Courier', 10), borderwidth=1, focusthickness=3)
        self.style.configure('TEntry', fieldbackground='#333333', foreground='#ff8800', 
                           font=('Courier', 10))
        self.style.configure('TLabelframe', background='#1a1a1a', foreground='#ff8800')
        self.style.configure('TLabelframe.Label', background='#1a1a1a', foreground='#ff8800')
        
        self.style.configure('TScrollbar', background='#333333', troughcolor='#1a1a1a')
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        title_label = ttk.Label(main_frame, text="Пепе", 
                               font=("Courier", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Кнопка смены профиля
        profile_btn = ttk.Button(main_frame, text="👤 Сменить профиль", command=self.show_profile_selection)
        profile_btn.grid(row=0, column=1, sticky=tk.E, pady=(0, 10))
        
        clear_btn = ttk.Button(main_frame, text="🧹 Очистить историю", command=self.clear_history)
        clear_btn.grid(row=0, column=2, sticky=tk.E, pady=(0, 10))
        
        search_frame = ttk.Frame(main_frame)
        search_frame.grid(row=0, column=3, sticky=tk.E, pady=(0, 10))
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.grid(row=0, column=0, padx=(0, 5))
        search_entry.bind("<Return>", lambda event: self.search_history())
        
        search_btn = ttk.Button(search_frame, text="🔍 Поиск", command=self.search_history)
        search_btn.grid(row=0, column=1)
        
        history_frame = ttk.LabelFrame(main_frame, text="История диалога", padding="5")
        history_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
        self.history_text = scrolledtext.ScrolledText(
            history_frame, 
            width=70, 
            height=20, 
            state=tk.DISABLED,
            bg='#222222',
            fg='#ff8800',
            insertbackground='#ff8800',
            selectbackground='#555555',
            font=('Courier', 10),
            relief='flat',
            borderwidth=2
        )
        self.history_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        self.input_field = ttk.Entry(input_frame, width=60)
        self.input_field.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.input_field.bind("<Return>", lambda event: self.process_input())
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=1, sticky=(tk.E,), pady=(0, 10))
        
        self.send_button = ttk.Button(button_frame, text="Отправить", command=self.process_input)
        self.send_button.grid(row=0, column=0, padx=(5, 0))
        
        self.voice_button = ttk.Button(button_frame, text="🎤 Голосовой ввод", command=self.start_voice_input)
        self.voice_button.grid(row=0, column=1, padx=(5, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("Готова к работе")
        status_bar = ttk.Label(
            main_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            background='#333333',
            foreground='#ff8800',
            font=('Courier', 9)
        )
        status_bar.grid(row=3, column=0, columnspan=4, sticky=(tk.W, tk.E))
        
        self.load_history_to_ui()
    
    def update_status(self, message):
        """Обновление статуса в GUI"""
        self.status_var.set(message)
    
    def search_history(self):
        query = self.search_var.get().strip()
        if not query:
            return
            
        self.status_var.set(f"Поиск: {query}...")
        Thread(target=self.search_history_async, args=(query,), daemon=True).start()
    
    def search_history_async(self, query):
        try:
            results = self.assistant.search_in_history(query, 10)
            
            self.root.after(0, self.show_search_results, query, results)
            self.root.after(0, lambda: self.status_var.set(f"Найдено {len(results)} результатов"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Ошибка поиска: {str(e)}"))
    
    def show_search_results(self, query, results):
        result_window = tk.Toplevel(self.root)
        result_window.title(f"Результаты поиска: {query}")
        result_window.geometry("800x400")
        result_window.configure(bg='#1a1a1a')
        
        result_text = scrolledtext.ScrolledText(
            result_window,
            wrap=tk.WORD,
            bg='#222222',
            fg='#ff8800',
            font=('Courier', 10),
            relief='flat'
        )
        result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        if results:
            for role, message, timestamp in results:
                role_icon = "👤" if role == "Пользователь" else "🤖"
                result_text.insert(tk.END, f"{timestamp} {role_icon} {role}: {message}\n\n")
        else:
            result_text.insert(tk.END, "Ничего не найдено")
            
        result_text.configure(state=tk.DISABLED)
        
        close_btn = ttk.Button(
            result_window,
            text="Закрыть",
            command=result_window.destroy
        )
        close_btn.pack(pady=10)

    def load_history_to_ui(self):
        self.history_text.configure(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        
        for dialog in self.assistant.dialog_history:
            role_icon = "👤" if dialog["role"] == "Пользователь" else "🤖"
            timestamp = dialog.get("timestamp", "")
            self.history_text.insert(tk.END, f"{timestamp} {role_icon} {dialog['role']}: {dialog['message']}\n\n")
        
        self.history_text.see(tk.END)
        self.history_text.configure(state=tk.DISABLED)
    
    def clear_history(self):
        self.assistant.dialog_history = []
        self.assistant.save_dialog_history()
        self.load_history_to_ui()
        self.status_var.set("История диалогов очищена")
    
    def update_history(self, role, text):
        self.history_text.configure(state=tk.NORMAL)
        role_icon = "👤" if role == "Пользователь" else "🤖"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.history_text.insert(tk.END, f"{timestamp} {role_icon} {role}: {text}\n\n")
        self.history_text.see(tk.END)
        self.history_text.configure(state=tk.DISABLED)
    
    def process_input(self):
        if self.is_generating:
            return
            
        text = self.input_field.get().strip()
        if not text:
            return
            
        self.input_field.delete(0, tk.END)
        self.update_history("Пользователь", text)
        self.is_generating = True
        
        # Создаем временное сообщение о генерации
        self.update_history("Пепе", "🤔 Думаю...")
        
        Thread(target=self.process_input_async, args=(text,), daemon=True).start()
    
    def process_input_async(self, text):
        try:
            response = self.assistant.generate_answer(text, self.update_status)
        
            if response:
                # Удаляем временное сообщение и добавляем полный ответ
                self.history_text.configure(state=tk.NORMAL)
                self.history_text.delete("end-2l", "end-1l")
                self.history_text.see(tk.END)
                self.history_text.configure(state=tk.DISABLED)
                
                self.root.after(0, self.update_history, "Пеpe", response)
                self.assistant.speak_text(response)
            
            self.root.after(0, lambda: self.status_var.set("Готова к работе"))
            self.is_generating = False
            
        except Exception as error:
            error_msg = f"Ошибка: {str(error)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            fallback = "Что-то не так с моими мозгами. Позовите моего Отца."
            self.root.after(0, self.update_history, "Пеpe", fallback)
            self.assistant.speak_text(fallback)
            self.is_generating = False

    def start_voice_input(self):
        if self.is_generating:
            return
            
        self.status_var.set("Говорите... (запись в течение 10 секунд)")
        Thread(target=self.voice_input_async, daemon=True).start()
    
    def voice_input_async(self):
        try:
            recognized_text = self.assistant.listen_and_recognize()
            if recognized_text:
                self.root.after(0, self.update_history, "Пользователь", recognized_text)
                self.root.after(0, lambda: self.status_var.set("Генерация ответа..."))
                
                # Создаем временное сообщение о генерации
                self.update_history("Пеpe", "🤔 Думаю...")
                self.is_generating = True
                
                response = self.assistant.process_text_input(recognized_text, self.update_status)
                if response:
                    # Удаляем временное сообщение и добавляем полный ответ
                    self.history_text.configure(state=tk.NORMAL)
                    self.history_text.delete("end-2l", "end-1l")
                    self.history_text.see(tk.END)
                    self.history_text.configure(state=tk.DISABLED)
                    
                    self.root.after(0, self.update_history, "Пеpe", response)
            
            self.root.after(0, lambda: self.status_var.set("Готова к работе"))
            self.is_generating = False
            
        except Exception as error:
            self.root.after(0, lambda e=error: self.status_var.set(f"Ошибка: {str(e)}"))
            self.is_generating = False
    
    def run(self):
        self.root.mainloop()

def main():
    assistant = VoiceAssistant()
    gui = AssistantGUI(assistant)
    gui.run()

if __name__ == "__main__":
    main()