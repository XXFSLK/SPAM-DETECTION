# the newest version in testGUI.py


import tkinter as tk
import os
from tkinter import messagebox, scrolledtext
import subprocess
import threading

class TextClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Classification System")
        # 设置工作目录
        os.chdir("F:/Graduation Design/Graduation Design/Tencent/NeuralNLP-NeuralClassifier")
        # Train button
        self.train_button = tk.Button(root, text="Train Model", command=self.open_train_window)
        self.train_button.pack(pady=10)

        # Evaluate button
        self.eval_button = tk.Button(root, text="Evaluate Model", command=self.open_eval_window)
        self.eval_button.pack(pady=10)

        # Predict button
        self.predict_button = tk.Button(root, text="Tag Text", command=self.open_predict_window)
        self.predict_button.pack(pady=10)

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
            command = ["python", "train.py", "conf/train.hmcn.json"]
            
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
            command = ["python", "eval.py", "conf/train.hmcn.json"]
            
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
        """Open a new window for tagging text."""
        self.predict_window = tk.Toplevel(self.root)
        self.predict_window.title("Tag Text")
        self.text_area = scrolledtext.ScrolledText(self.predict_window, width=80, height=20)
        self.text_area.pack(pady=10)
        self.tag_button = tk.Button(self.predict_window, text="Tag Text", command=self.tag_text)
        self.tag_button.pack(pady=10)

    def tag_text(self):
        """Process the text input for tagging."""
        text = self.text_area.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please enter text to tag.")
            return

        # Here you would call your predict function to tag the text
        # For example:
        # labels = classify_content(text, model)  # Assuming classify_content is defined
        # Display the results
        # self.text_area.insert(tk.END, f"Tagged Text: {labels}\n")

        # Placeholder for demonstration
        self.text_area.insert(tk.END, f"Tagged Text: [Placeholder for labels]\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextClassificationApp(root)
    root.mainloop()
