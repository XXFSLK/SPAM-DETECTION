import tkinter as tk
import os
from tkinter import messagebox, scrolledtext, filedialog
import subprocess
import threading
import re
import jieba
import json
import chardet

class TextClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Classification System")
        os.chdir("F:/Graduation Design/Graduation Design/Tencent/NeuralNLP-NeuralClassifier")
        
        self.predict_json_path = os.path.join("data", "predict.json")
        
        # 按钮区域
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)
        
        self.train_button = tk.Button(button_frame, text="Train Model", command=self.open_train_window)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.eval_button = tk.Button(button_frame, text="Evaluate Model", command=self.open_eval_window)
        self.eval_button.pack(side=tk.LEFT, padx=5)
        
        self.predict_button = tk.Button(button_frame, text="Predict", command=self.open_predict_window)
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # 新增运行按钮
        self.run_button = tk.Button(button_frame, text="Run Prediction", 
                                  command=self.run_prediction,
                                  state=tk.DISABLED)  # 初始不可用state=tk.DISABLED
        self.run_button.pack(side=tk.LEFT, padx=5)

    def open_train_window(self):
        """Open a new window for training."""
        self.train_window = tk.Toplevel(self.root)
        self.train_window.title("Training Model")
        self.train_text_area = scrolledtext.ScrolledText(self.train_window, width=80, height=20)
        self.train_text_area.pack(pady=10)
        self.start_training_button = tk.Button(self.train_window, text="Start Training", command=self.start_training)
        self.start_training_button.pack(pady=10)

    def start_training(self):
        """Start training in a separate thread."""
        self.train_text_area.delete('1.0', tk.END)  # Clear the text area
        threading.Thread(target=self.run_training).start()

    def run_training(self):
        """Run the training process and update the text area."""
        try:
            self.train_text_area.insert(tk.END, "Starting training...\n")
            self.train_text_area.yview(tk.END)
            
            # 构造命令
            command = ["python", "train.py", "conf/train.hmcnRec06.json"]
            
            # 启动训练进程
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # 读取输出并实时更新文本区域
            for line in iter(process.stdout.readline, ''):
                self.train_text_area.insert(tk.END, line)
                self.train_text_area.yview(tk.END)  # 滚动到文本区域的末尾
            
            # 等待进程结束
            process.stdout.close()
            process.wait()
            
            self.train_text_area.insert(tk.END, "Training completed.\n")
        except Exception as e:
            self.train_text_area.insert(tk.END, f"Error starting training: {str(e)}\n")

    def open_eval_window(self):
        """Open a new window for evaluation."""
        self.eval_window = tk.Toplevel(self.root)
        self.eval_window.title("Evaluate Model")
        self.eval_text_area = scrolledtext.ScrolledText(self.eval_window, width=80, height=20)
        self.eval_text_area.pack(pady=10)
        self.start_eval_button = tk.Button(self.eval_window, text="Start Evaluation", command=self.start_evaluation)
        self.start_eval_button.pack(pady=10)

    def start_evaluation(self):
        """Start evaluation in a separate thread."""
        self.eval_text_area.delete('1.0', tk.END)
        threading.Thread(target=self.run_evaluation).start()

    def run_evaluation(self):
        """Run the evaluation process and update the text area."""
        try:
            self.eval_text_area.insert(tk.END, "Starting evaluation...\n")
            self.eval_text_area.yview(tk.END)
            
            # 构造命令
            command = ["python", "eval.py", "conf/train.hmcnRec06.json"]
            
            # 启动评估进程
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # 读取输出并实时更新文本区域
            for line in iter(process.stdout.readline, ''):
                self.eval_text_area.insert(tk.END, line)
                self.eval_text_area.yview(tk.END)  # 滚动到文本区域的末尾
            
            # 等待进程结束
            process.stdout.close()
            process.wait()
            
            self.eval_text_area.insert(tk.END, "Evaluation completed.\n")
        except Exception as e:
            self.eval_text_area.insert(tk.END, f"Error starting evaluation: {str(e)}\n")


    def open_predict_window(self):
        """预测窗口：支持文本输入或批量文件选择"""
        self.predict_window = tk.Toplevel(self.root)
        self.predict_window.title("Text Prediction")
        
        # 文件选择按钮（支持多选）
        self.select_file_btn = tk.Button(
            self.predict_window, 
            text="Select TXT File(s)", 
            command=lambda: self.load_txt_files(batch_mode=True)
        )
        self.select_file_btn.pack(pady=5)
        
        # 文本输入区域
        self.text_area = scrolledtext.ScrolledText(self.predict_window, width=80, height=20)
        self.text_area.pack(pady=10)
        
        # 处理按钮
        self.process_btn = tk.Button(
            self.predict_window, 
            text="Process & Save JSON", 
            command=self.process_text
        )
        self.process_btn.pack(pady=10)

    def load_txt_files(self, batch_mode=False):
        """加载TXT文件到处理队列"""
        file_paths = filedialog.askopenfilenames(filetypes=[("Text Files", "*.txt")])
        if not file_paths: return
        
        if batch_mode:
            self.batch_files = file_paths  # 存储批量文件路径
            self.text_area.delete('1.0', tk.END)
            self.text_area.insert(tk.END, f"已选择 {len(file_paths)} 个文件:\n" + "\n".join(file_paths))
        else:
            # 单个文件加载到文本区域（原逻辑）
            encoding = self.detect_encoding(file_paths[0])
            with open(file_paths[0], 'r', encoding=encoding, errors='ignore') as f:
                self.text_area.delete('1.0', tk.END)
                self.text_area.insert(tk.END, f.read())

    @staticmethod
    def detect_encoding(file_path):
        """动态检测文件编码"""
        with open(file_path, 'rb') as f:
            return chardet.detect(f.read())['encoding']

    def process_text(self):
        """处理文本并保存为JSON（支持批量）"""
        # 判断处理模式
        if hasattr(self, 'batch_files'):
            self._process_batch_files()
        else:
            self._process_single_text()

    def _process_single_text(self):
        """处理单个文本输入"""
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please input text or select files.")
            return
        
        tokens = self.preprocess_text(text)
        output_data = [{
            "doc_label": [],
            "doc_token": tokens,
            "doc_keyword": [],
            "doc_topic": []
        }]
        
        self._save_json(output_data)

    def _process_batch_files(self):
        """批量处理多个文件"""
        processed_data = []
        for file_path in self.batch_files:
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                text = f.read()
            
            tokens = self.preprocess_text(text)
            processed_data.append({
                "doc_label": [],
                "doc_token": tokens,
                "doc_keyword": [],
                "doc_topic": []
            })
        
        self._save_json(processed_data)

    def _save_json(self, data):
        """统一保存JSON文件（单行格式）"""
        output_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if not output_path: return
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        messagebox.showinfo("Success", f"已保存预处理数据至：\n{output_path}")
        delattr(self, 'batch_files', None)  # 清除批量文件缓存

    @staticmethod
    def preprocess_text(text):
        """文本预处理：分词+过滤"""
        tokens = list(jieba.cut(text))
        return [token for token in tokens if re.match(r'^[\u4e00-\u9fa5a-zA-Z]+$', token)]
    def _save_json(self, data):
        """保存JSON文件并启用运行按钮"""
        # 确保data目录存在
        os.makedirs("data", exist_ok=True)
        
        # 保存到固定路径
        self.predict_json_path = os.path.join("data", "predict.json")
        with open(self.predict_json_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # 启用运行按钮
        self.run_button.config(state=tk.NORMAL)
        messagebox.showinfo("Success", f"预测文件已保存至：\n{self.predict_json_path}")

    def run_prediction(self):
        """执行预测命令"""
        # 创建新窗口显示输出
        self.run_window = tk.Toplevel(self.root)
        self.run_window.title("Prediction Output")
        
        # 输出文本区域
        self.output_area = scrolledtext.ScrolledText(self.run_window, width=80, height=20)
        self.output_area.pack(pady=10)
        
        # 在独立线程中运行命令
        threading.Thread(target=self._execute_prediction).start()

    def _execute_prediction(self):
        """实际执行预测命令"""
        try:
            # 构造命令
            command = [
                "python", "predict.py",
                "conf/train.hmcnRec06.json",
                self.predict_json_path
            ]
            
            # 执行命令
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='GBK'   #针对中文文本，后期可以考虑编写动态识别编码
            )
            
            # 实时显示输出
            for line in iter(process.stdout.readline, ''):
                self.output_area.insert(tk.END, line)
                self.output_area.yview(tk.END)
            
            process.stdout.close()
            return_code = process.wait()
            
            if return_code == 0:
                self.output_area.insert(tk.END, "\nPrediction completed successfully!\n")
            else:
                self.output_area.insert(tk.END, f"\nPrediction failed with code {return_code}\n")
                
        except Exception as e:
            self.output_area.insert(tk.END, f"Error: {str(e)}\n")
        finally:
            self.run_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = TextClassificationApp(root)
    root.mainloop()