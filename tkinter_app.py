import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk  # Nécessite Pillow pour gérer les images
import threading
from data_collector import collect_landmarks
import modeling
import model_tester
import warnings
import csv
import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore")

class FaceAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Analysis Application")
        self.root.geometry("600x600")
        self.TitleTEXT = ("Helvetica", 16, "bold")
        self.ButtonFont = ("Helvetica", 12, "bold")
        # Définir l'icône de l'application
        try:
            # Charger l'image avec Pillow
            icon_image = Image.open("logo.png")  # Remplacez 'icon.png' par le chemin de votre fichier .png ou .jpg
            icon_photo = ImageTk.PhotoImage(icon_image)
            self.root.iconphoto(True, icon_photo)
            # Conserver une référence pour éviter que l'image ne soit supprimée par le garbage collector
            self.icon_photo = icon_photo
        except Exception as e:
            logging.error(f"Erreur lors du chargement de l'icône : {str(e)}")
            messagebox.showwarning("Avertissement", "Impossible de charger l'icône de l'application.")
        # Configure root with a solid background to avoid transparency issues
        self.root.configure(bg="#f0f4f8")
        self.create_gradient_canvas()

        # Create frames with a matching background
        self.home_frame = tk.Frame(self.root, bg="#f0f4f8")
        self.data_frame = tk.Frame(self.root, bg="#f0f4f8")
        self.manage_classes_frame = tk.Frame(self.root, bg="#f0f4f8")
        self.user_management_frame = tk.Frame(self.root, bg="#f0f4f8")

        # Initialize home interface with background image
        self.setup_home_frame()

        # Initially show home frame
        self.home_frame.pack(fill="both", expand=True)

        # Bind resize event to update background image
        self.root.bind("<Configure>", self.resize_background)

    def create_gradient_canvas(self):
        # Create a canvas for gradient background, placed behind all widgets
        self.canvas = tk.Canvas(self.root, width=600, height=600, highlightthickness=0, bg="#f0f4f8")
        self.canvas.place(x=0, y=0)
        # Create gradient from top (#4e73df) to bottom (#224abe)
        for i in range(600):
            r = int(78 + (34 - 78) * i / 600)
            g = int(115 + (74 - 115) * i / 600)
            b = int(223 + (190 - 223) * i / 600)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas.create_line(0, i, 600, i, fill=color)

    def setup_home_frame(self):
        # Create a canvas for the home frame to support the background image
        self.home_canvas = tk.Canvas(self.home_frame, highlightthickness=0, bg="#f0f4f8")
        self.home_canvas.pack(fill="both", expand=True)

        # Load and set background image
        try:
            # Charger l'image avec Pillow (remplacez 'background.png' par le chemin de votre image)
            self.bg_image = Image.open("background.png")  # Utilisez .jpg ou .png selon votre fichier
            self.update_background_image()  # Initial resize and display
        except Exception as e:
            logging.error(f"Erreur lors du chargement de l'image de fond : {str(e)}")
            messagebox.showwarning("Avertissement", "Impossible de charger l'image de fond. Utilisation du gradient par défaut.")

        # Main window title with shadow effect
        self.title_label = tk.Label(
            self.home_frame, 
            text="Face Tracking Application", 
            font=("Comic Sans MS", 20, "bold"),
            fg="#ffffff",
            bg="#385280",
            pady=10
        )
        self.home_canvas.create_window(300, 50, window=self.title_label, tags="widget")

        # Button style configuration
        button_style = {
            "font": self.ButtonFont,
            "width": 18,
            "height": 2,
            "relief": "flat",
            "cursor": "hand2"
        }

        # Collect Data button with icon-like styling
        self.collect_data_button = tk.Button(
            self.home_frame, 
            text="📷 Collect Data", 
            command=self.show_data_collection,
            bg="#25194F",
            fg="white",
            **button_style
        )
        self.home_canvas.create_window(300, 150, window=self.collect_data_button, tags="widget")

        # Train Model button
        self.train_model_button = tk.Button(
            self.home_frame, 
            text="🧠 Train Model", 
            command=self.train_model,
            bg="#1E1540",
            fg="white",
            **button_style
        )
        self.home_canvas.create_window(300, 250, window=self.train_model_button, tags="widget")

        # Test Model button
        self.test_model_button = tk.Button(
            self.home_frame, 
            text="🔍 Test Model", 
            command=self.test_model,
            bg="#1A1239",
            fg="white",
            **button_style
        )
        self.home_canvas.create_window(300, 350, window=self.test_model_button, tags="widget")

        # Manage Classes button
        self.manage_classes_button = tk.Button(
            self.home_frame, 
            text="📋 Manage Classes", 
            command=self.show_manage_classes,
            bg="#333544",
            fg="white",
            **button_style
        )
        self.home_canvas.create_window(300, 450, window=self.manage_classes_button, tags="widget")

        # User Management button
        self.user_management_button = tk.Button(
            self.home_frame, 
            text="👤 User Management", 
            command=self.show_user_management,
            bg="#2C2D3A",
            fg="white",
            **button_style
        )
        self.home_canvas.create_window(300, 550, window=self.user_management_button, tags="widget")

    def update_background_image(self, event=None):
        # Get current canvas size
        canvas_width = self.home_canvas.winfo_width()
        canvas_height = self.home_canvas.winfo_height()

        # Only update if canvas has a valid size
        if canvas_width <= 1 or canvas_height <= 1:
            return

        try:
            # Resize image to match canvas size
            resized_image = self.bg_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(resized_image)
            # Update or create background image
            if hasattr(self, 'bg_image_id'):
                self.home_canvas.itemconfig(self.bg_image_id, image=self.bg_photo)
            else:
                self.bg_image_id = self.home_canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
            # Ensure background image is behind all widgets
            self.home_canvas.lower(self.bg_image_id)
        except Exception as e:
            logging.error(f"Erreur lors du redimensionnement de l'image de fond : {str(e)}")

    def resize_background(self, event):
        # Only update if the home frame is visible
        if self.home_frame.winfo_ismapped():
            self.update_background_image()

    def train_model(self):
        try:
            modeling.train_and_evaluate_models()
            messagebox.showinfo("Success", "Model training completed. Check the project directory for metrics and plots.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")

    def test_model(self):
        try:
            self.model_tester = model_tester.ModelTester(self.root)  # Pass the root to manage updates
            self.model_tester.start()  # Start the real-time testing
            messagebox.showinfo("Info", "Model testing started. Check the webcam feed. Press 'q' to stop.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to test model: {str(e)}")

    def setup_data_frame(self):
        self.data_collector = DataCollector(self.data_frame)
        self.return_button = tk.Button(
            self.data_frame,
            text="← Back",
            command=self.show_home,
            bg="#858796",
            fg="white",
            font=self.ButtonFont,
            width=10,
            relief="flat",
            cursor="hand2"
        )
        self.return_button.place(x=10, y=10)

    def setup_manage_classes_frame(self):
        for widget in self.manage_classes_frame.winfo_children():
            widget.destroy()

        return_button = tk.Button(
            self.manage_classes_frame,
            text="← Back",
            command=self.show_home,
            bg="#858796",
            fg="white",
            font=self.ButtonFont,
            width=10,
            relief="flat",
            cursor="hand2"
        )
        return_button.place(x=10, y=10)

        title_label = tk.Label(
            self.manage_classes_frame,
            text="Manage Classes",
            font=self.TitleTEXT,
            fg="#ffffff",
            bg="#697cb3"
        )
        title_label.pack(pady=20)

        try:
            with open('class_descriptions.csv', mode='r', newline='') as f:
                csv_reader = csv.reader(f)
                next(csv_reader, None)
                classes = [(row[0], row[1]) for row in csv_reader if len(row) >= 2]
        except FileNotFoundError:
            classes = []
            tk.Label(
                self.manage_classes_frame,
                text="No classes found in class_descriptions.csv",
                font=("Helvetica", 10, "italic"),
                fg="#e74a3b",
                bg="#f0f4f8"
            ).pack(pady=10)

        for class_name, description in classes:
            frame = tk.Frame(self.manage_classes_frame, bg="#ffffff", bd=1, relief="solid")
            frame.pack(fill="x", padx=20, pady=5)
            tk.Label(
                frame,
                text=f"Class: {class_name} - Description: {description}",
                font=("Helvetica", 10),
                anchor="w",
                bg="#ffffff",
                padx=10,
                pady=5
            ).pack(side="left", fill="x", expand=True)
            tk.Button(
                frame,
                text="Delete",
                command=lambda cn=class_name: self.delete_class(cn),
                bg="#e74a3b",
                fg="white",
                font=("Helvetica", 10),
                relief="flat",
                cursor="hand2"
            ).pack(side="right", padx=5)

        if not classes:
            tk.Label(
                self.manage_classes_frame,
                text="No classes available.",
                font=("Helvetica", 10, "italic"),
                fg="#ffffff",
                bg="#f0f4f8"
            ).pack(pady=10)

    def delete_class(self, class_name):
        try:
            temp_file = 'class_descriptions_temp.csv'
            with open('class_descriptions.csv', mode='r', newline='') as f, \
                 open(temp_file, mode='w', newline='') as temp:
                csv_reader = csv.reader(f)
                csv_writer = csv.writer(temp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                header = next(csv_reader)
                csv_writer.writerow(header)
                for row in csv_reader:
                    if len(row) >= 1 and row[0] != class_name:
                        csv_writer.writerow(row)
            os.replace(temp_file, 'class_descriptions.csv')

            try:
                df = pd.read_csv('coords.csv', header=0)
                df = df[df.iloc[:, 0] != class_name]
                if df.empty:
                    with open('coords.csv', mode='w', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(['class'] + [f'face_{i}_{attr}' for i in range(468) for attr in ['x', 'y', 'z', 'visibility']])
                else:
                    df.to_csv('coords.csv', index=False)
            except FileNotFoundError:
                with open('coords.csv', mode='w', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(['class'] + [f'face_{i}_{attr}' for i in range(468) for attr in ['x', 'y', 'z', 'visibility']])

            if not df.empty:
                modeling.train_and_evaluate_models()
                messagebox.showinfo("Success", f"Class '{class_name}' deleted and model retrained.")
            else:
                messagebox.showinfo("Success", f"Class '{class_name}' deleted. No data left in coords.csv, so model was not retrained.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete class or retrain model: {str(e)}")
        self.setup_manage_classes_frame()

    def show_data_collection(self):
        self.home_frame.pack_forget()
        self.user_management_frame.pack_forget()
        if not hasattr(self, 'data_collector'):
            self.setup_data_frame()
        self.data_frame.pack(fill="both", expand=True)

    def show_manage_classes(self):
        self.home_frame.pack_forget()
        self.user_management_frame.pack_forget()
        self.setup_manage_classes_frame()
        self.manage_classes_frame.pack(fill="both", expand=True)

    def show_user_management(self):
        logging.debug("Loading GUI page: User Management")
        self.home_frame.pack_forget()
        self.data_frame.pack_forget()
        self.manage_classes_frame.pack_forget()
        self.setup_user_management_frame()
        self.user_management_frame.pack(fill="both", expand=True)

    def setup_user_management_frame(self):
        for widget in self.user_management_frame.winfo_children():
            widget.destroy()

        try:
            label = tk.Label(
                self.user_management_frame, 
                text="User Management", 
                font=self.TitleTEXT,
                fg="#ffffff",
                bg="#09194a"
            )
            label.pack(pady=(20, 5))

            button_frame = tk.Frame(self.user_management_frame, bg="#f0f4f8")
            button_frame.pack(pady=10)

            button_width_narrow = 20
            button_style = {
                "font": self.ButtonFont,
                "width": button_width_narrow,
                "relief": "flat",
                "cursor": "hand2"
            }

            add_user_button = tk.Button(
                button_frame, 
                text="➕ Add User", 
                command=self.create_add_user_view, 
                bg="#1cc88a",
                fg="white",
                **button_style
            )
            add_user_button.pack(pady=10)

            delete_user_button = tk.Button(
                button_frame, 
                text="🗑 Delete User", 
                command=self.create_delete_user_view, 
                bg="#e74a3b",
                fg="white",
                **button_style
            )
            delete_user_button.pack(pady=10)

            show_users_button = tk.Button(
                button_frame, 
                text="📋 Show Users", 
                command=self.create_show_users_view, 
                bg="#36b9cc",
                fg="white",
                **button_style
            )
            show_users_button.pack(pady=10)

            back_button = tk.Button(
                self.user_management_frame, 
                text="← Back", 
                command=self.show_home, 
                bg="#858796",
                fg="white",
                font=self.ButtonFont,
                relief="flat",
                cursor="hand2"
            )
            back_button.pack(pady=20)

            logging.debug("GUI page (User Management) loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load GUI page (User Management): {str(e)}")

    def create_add_user_view(self):
        logging.debug("Loading GUI page: Add User")
        for widget in self.user_management_frame.winfo_children():
            widget.destroy()

        try:
            label = tk.Label(
                self.user_management_frame, 
                text="Add User", 
                font=self.TitleTEXT,
                fg="#ffffff",
                bg="#4e73df"
            )
            label.pack(pady=(20, 5))

            self.username_entry = tk.Entry(
                self.user_management_frame, 
                width=30, 
                font=("Helvetica", 14),
                relief="flat",
                bg="#ffffff"
            )
            self.username_entry.pack(pady=10)

            confirm_button = tk.Button(
                self.user_management_frame, 
                text="✔ Confirm Input", 
                command=self.confirm_add_user_action, 
                bg="#1cc88a",
                fg="white",
                font=self.ButtonFont,
                relief="flat",
                cursor="hand2"
            )
            confirm_button.pack(pady=10)

            back_button = tk.Button(
                self.user_management_frame, 
                text="← Back", 
                command=self.show_user_management, 
                bg="#858796",
                fg="white",
                font=self.ButtonFont,
                relief="flat",
                cursor="hand2"
            )
            back_button.pack(pady=20)

            logging.debug("GUI page (Add User) loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load GUI page (Add User): {str(e)}")

    def confirm_add_user_action(self):
        username = self.username_entry.get().strip()
        if not username:
            messagebox.showerror("Error", "Please enter a username.")
            return

        profile_dir = os.path.join(os.getcwd(), "Profile")
        os.makedirs(profile_dir, exist_ok=True)
        user_dir = os.path.join(profile_dir, username)

        try:
            if os.path.exists(user_dir):
                messagebox.showerror("Error", f"User '{username}' already exists.")
            else:
                os.makedirs(user_dir)
                messagebox.showinfo("Success", f"User '{username}' added successfully.")
                logging.debug(f"User '{username}' created in Profile directory")
                self.show_user_management()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add user: {str(e)}")
            logging.error(f"Failed to create user '{username}': {str(e)}")

    def create_delete_user_view(self):
        logging.debug("Loading GUI page: Delete User")
        for widget in self.user_management_frame.winfo_children():
            widget.destroy()

        try:
            label = tk.Label(
                self.user_management_frame, 
                text="Delete User", 
                font=self.TitleTEXT,
                fg="#ffffff",
                bg="#4e73df"
            )
            label.pack(pady=(20, 5))

            self.user_var = tk.StringVar()
            self.user_dropdown = ttk.Combobox(
                self.user_management_frame, 
                textvariable=self.user_var, 
                font=("Helvetica", 14), 
                state="readonly",
                style="Custom.TCombobox"
            )
            self.user_dropdown.pack(pady=10)

            # Configure Combobox style
            style = ttk.Style()
            style.configure("Custom.TCombobox", fieldbackground="#ffffff")

            self.load_user_list()

            delete_button = tk.Button(
                self.user_management_frame, 
                text="🗑 Confirm Delete", 
                command=self.confirm_delete_user_action, 
                bg="#e74a3b",
                fg="white",
                font=self.ButtonFont,
                relief="flat",
                cursor="hand2"
            )
            delete_button.pack(pady=10)

            back_button = tk.Button(
                self.user_management_frame, 
                text="← Back", 
                command=self.show_user_management, 
                bg="#858796",
                fg="white",
                font=self.ButtonFont,
                relief="flat",
                cursor="hand2"
            )
            back_button.pack(pady=20)

            logging.debug("GUI page (Delete User) loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load GUI page (Delete User): {str(e)}")

    def load_user_list(self):
        profile_dir = os.path.join(os.getcwd(), "Profile")
        try:
            if os.path.exists(profile_dir):
                users = [d for d in os.listdir(profile_dir) if os.path.isdir(os.path.join(profile_dir, d))]
                self.user_dropdown['values'] = users
                if users:
                    self.user_dropdown.set(users[0])
                else:
                    self.user_dropdown['values'] = ["No users available"]
                    self.user_dropdown.set("No users available")
            else:
                self.user_dropdown['values'] = ["No users available"]
                self.user_dropdown.set("No users available")
        except Exception as e:
            logging.error(f"Failed to load user list: {str(e)}")

    def confirm_delete_user_action(self):
        username = self.user_var.get()
        if not username or username == "No users available":
            messagebox.showerror("Error", "Please select a valid user to delete.")
            return

        profile_dir = os.path.join(os.getcwd(), "Profile")
        user_dir = os.path.join(profile_dir, username)

        try:
            if os.path.exists(user_dir):
                import shutil
                shutil.rmtree(user_dir)
                messagebox.showinfo("Success", f"User '{username}' deleted successfully.")
                logging.debug(f"User '{username}' deleted from Profile directory")
                self.load_user_list()
            else:
                messagebox.showerror("Error", f"User '{username}' does not exist.")
            self.show_user_management()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete user: {str(e)}")
            logging.error(f"Failed to delete user '{username}': {str(e)}")

    def create_show_users_view(self):
        logging.debug("Loading GUI page: Show Users")
        for widget in self.user_management_frame.winfo_children():
            widget.destroy()

        try:
            label = tk.Label(
                self.user_management_frame, 
                text="Show Users", 
                font=self.TitleTEXT,
                fg="#ffffff",
                bg="#4e73df"
            )
            label.pack(pady=(20, 5))

            self.users_listbox = tk.Listbox(
                self.user_management_frame, 
                font=("Helvetica", 12), 
                width=40, 
                height=10,
                bg="#ffffff",
                relief="flat"
            )
            self.users_listbox.pack(pady=10)

            self.load_users_into_listbox()

            confirm_button = tk.Button(
                self.user_management_frame, 
                text="🔄 Refresh Users", 
                command=self.load_users_into_listbox, 
                bg="#36b9cc",
                fg="white",
                font=self.ButtonFont,
                relief="flat",
                cursor="hand2"
            )
            confirm_button.pack(pady=10)

            back_button = tk.Button(
                self.user_management_frame, 
                text="← Back", 
                command=self.show_user_management, 
                bg="#858796",
                fg="white",
                font=self.ButtonFont,
                relief="flat",
                cursor="hand2"
            )
            back_button.pack(pady=20)

            logging.debug("GUI page (Show Users) loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load GUI page (Show Users): {str(e)}")

    def load_users_into_listbox(self):
        self.users_listbox.delete(0, tk.END)
        profile_dir = os.path.join(os.getcwd(), "Profile")
        try:
            if os.path.exists(profile_dir):
                users = [d for d in os.listdir(profile_dir) if os.path.isdir(os.path.join(profile_dir, d))]
                for user in users:
                    self.users_listbox.insert(tk.END, user)
                if not users:
                    self.users_listbox.insert(tk.END, "No users available")
                logging.debug(f"Loaded users into listbox: {users}")
            else:
                self.users_listbox.insert(tk.END, "No users available")
                logging.warning("Profile directory does not exist.")
        except Exception as e:
            self.users_listbox.insert(tk.END, "Error loading users")
            logging.error(f"Failed to load users into listbox: {str(e)}")

    def show_home(self):
        if hasattr(self, 'data_collector') and hasattr(self.data_collector, 'is_collecting') and self.data_collector.is_collecting:
            self.data_collector.status_label.config(text="Status: Stopping (press 'q' to confirm)")
        self.data_frame.pack_forget()
        self.manage_classes_frame.pack_forget()
        self.user_management_frame.pack_forget()
        self.home_frame.pack(fill="both", expand=True)
        self.update_background_image()  # Update background when returning to home

class DataCollector:
    def __init__(self, frame):
        self.frame = frame
        self.is_collecting = False

        self.class_label = tk.Label(
            frame, 
            text="Class Name:", 
            font=("Helvetica", 12),
            fg="#ffffff",
            bg="#4C65B7"
        )
        self.class_label.pack(pady=10)
        self.class_entry = tk.Entry(
            frame, 
            width=30,
            font=("Helvetica", 12),
            relief="flat",
            bg="#ffffff"
        )
        self.class_entry.pack(pady=5)

        self.description_label = tk.Label(
            frame, 
            text="Description:", 
            font=("Helvetica", 12),
            fg="#ffffff",
            bg="#4C65B7"
        )
        self.description_label.pack(pady=10)
        self.description_entry = tk.Entry(
            frame, 
            width=30,
            font=("Helvetica", 12),
            relief="flat",
            bg="#ffffff"
        )
        self.description_entry.pack(pady=5)

        self.show_messages_var = tk.BooleanVar(value=True)
        self.message_checkbox = tk.Checkbutton(
            frame, 
            text="Show Confirmation Messages", 
            variable=self.show_messages_var,
            font=("Helvetica", 10),
            fg="#f0f4f8",
            bg="#f0f4f8",
            selectcolor="#283458"
        )
        self.message_checkbox.pack(pady=10)

        self.collect_button = tk.Button(
            frame, 
            text="▶ Start Collection", 
            command=self.toggle_collection,
            bg="#449577",
            fg="white",
            font=("Helvetica", 12, "bold"),
            width=15,
            relief="flat",
            cursor="hand2"
        )
        self.collect_button.pack(pady=10)

        self.status_label = tk.Label(
            frame, 
            text="Click Start", 
            font=("Helvetica", 10, "italic"),
            fg="#ffffff",
            bg="#474d61"
        )
        self.status_label.pack(pady=10)

        self.instructions = tk.Label(
            frame, 
            text="Enter a class name and description, then click 'Start Collection'.\nPress 'q' to stop camera.",
            font=("Helvetica", 10),
            fg="#ffffff",
            bg="#4C65B7",
            justify="center"
        )
        self.instructions.pack(pady=10)

    def save_class_description(self, class_name, description, csv_file='class_descriptions.csv'):
        try:
            with open(csv_file, mode='x', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['class_name', 'description'])
        except FileExistsError:
            pass

        try:
            with open(csv_file, mode='r', newline='') as f:
                csv_reader = csv.reader(f)
                next(csv_reader, None)
                existing_classes = [row[0] for row in csv_reader]
        except FileNotFoundError:
            existing_classes = []

        if class_name not in existing_classes:
            with open(csv_file, mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow([class_name, description])

    def toggle_collection(self):
        if not self.is_collecting:
            class_name = self.class_entry.get().strip()
            description = self.description_entry.get().strip()
            if not class_name:
                messagebox.showerror("Error", "Please enter a class name.")
                return
            if not description:
                messagebox.showerror("Error", "Please enter a description.")
                return

            self.save_class_description(class_name, description)

            self.is_collecting = True
            self.collect_button.config(text="⏹ Stop Collection", bg="#e74a3b")
            self.status_label.config(text=f"Status: Collecting for '{class_name}'")

            if self.show_messages_var.get():
                messagebox.showinfo("Started", f"Started collecting landmarks for class '{class_name}'.")

            self.collection_thread = threading.Thread(
                target=collect_landmarks, 
                args=(class_name,),
                daemon=True
            )
            self.collection_thread.start()

            self.frame.after(100, self.check_collection_status)
        else:
            self.status_label.config(text="Status: Stopping (press 'q' to confirm)")

    def check_collection_status(self):
        if self.is_collecting and not self.collection_thread.is_alive():
            self.is_collecting = False
            self.collect_button.config(text="▶ Start Collection", bg="#13533c")
            self.status_label.config(text="Status: Idle")
            if self.show_messages_var.get():
                messagebox.showinfo("Stopped", "Data collection stopped.")
        if self.is_collecting:
            self.frame.after(100, self.check_collection_status)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAnalysisApp(root)
    root.mainloop()