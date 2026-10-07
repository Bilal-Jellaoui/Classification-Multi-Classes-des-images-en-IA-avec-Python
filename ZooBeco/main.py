from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.clock import Clock, mainthread
from kivy.core.audio import SoundLoader
from kivy.properties import ObjectProperty, NumericProperty, StringProperty
from kivy.animation import Animation
from kivy import platform
from kivy.core.window import Window

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage

# tkinter uniquement pour desktop (sélection fichier)
import tkinter as tk
from tkinter import filedialog

# Taille fenêtre mobile
Window.size = (360, 640)

Builder.load_file('zoo.kv')

# Gestion ouverture galerie selon plateforme
if platform == 'android' or platform == 'ios':
    from plyer import filechooser

class GalleryManager:
    @staticmethod
    def open_gallery(callback):
        if platform in ('android', 'ios'):
            # Utilisation plyer.filechooser
            filechooser.open_file(on_selection=lambda selection: callback(selection[0] if selection else None))
        else:
            # Desktop : tkinter file dialog
            @mainthread
            def show_dialog():
                root = tk.Tk()
                root.withdraw()
                path = filedialog.askopenfilename(
                    title="Sélectionner une image",
                    filetypes=[("Images", "*.jpg *.jpeg *.png")]
                )
                root.destroy()
                callback(path if path else None)
            show_dialog()

class LoadingScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self.start_animation, 0.1)
    
    def start_animation(self, dt):
        anim = (Animation(opacity=1, duration=1.5) + 
               Animation(size_hint=(0.3, 0.3), duration=1.5))
        anim.start(self.ids.logo)
        Clock.schedule_once(self.switch_screen, 3)
    
    def switch_screen(self, dt):
        self.manager.current = 'welcome'

class WelcomeScreen(Screen):
    bg_music = ObjectProperty(None)
    
    def on_enter(self):
        self.bg_music = SoundLoader.load('assets/sounds/background_music.ogg')
        if self.bg_music:
            self.bg_music.loop = True
            self.bg_music.volume = 0.3
            self.bg_music.play()
    
    def on_leave(self):
        if self.bg_music:
            self.bg_music.stop()

class ClassificationScreen(Screen):
    result_text = StringProperty('')
    confidence_text = StringProperty('')
    loading_visible = NumericProperty(0)
    model = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.class_names = ['Buffalo', 'Chat', 'Chien', 'Éléphant', 'Oiseau', 'Zèbre']
        Clock.schedule_once(self.load_model, 0.5)
    
    def load_model(self, dt):
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'animal_classifierv7.keras')
            self.model = load_model(model_path)
        except Exception as e:
            self.result_text = "Erreur de chargement du modèle"
    
    def select_image(self):
        GalleryManager.open_gallery(self.handle_selection)
    
    def handle_selection(self, path):
        if path and os.path.exists(path):
            self.ids.img_preview.source = path
            self.result_text = ""
            self.confidence_text = ""
    
    def classify_image(self):
        if not self.ids.img_preview.source or not self.model:
            return
        
        self.loading_visible = 1
        self.result_text = ""
        self.confidence_text = ""
        
        try:
            img = PILImage.open(self.ids.img_preview.source).convert('RGB')
            img = img.resize((224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            Clock.schedule_once(lambda dt: self.run_prediction(img_array))
            SoundLoader.load('assets/sounds/click.wav').play()
        except Exception as e:
            self.handle_error(e)
    
    def run_prediction(self, img_array):
        try:
            predictions = self.model.predict(img_array)
            pred_class = np.argmax(predictions[0])
            confidence = predictions[0][pred_class] * 100
            
            if confidence < 70:
                self.result_text = "Inconnu"
                self.confidence_text = f"Confiance : {confidence:.2f}% "
            else:
                self.result_text = f"{self.class_names[pred_class]}"
                self.confidence_text = f"Confiance : {confidence:.2f}%"
                SoundLoader.load('assets/sounds/success.wav').play()
        except Exception as e:
            self.result_text = "Erreur de classification"
        finally:
            self.loading_visible = 0
    
    def handle_error(self, error):
        self.loading_visible = 0
        self.result_text = "Erreur de traitement"

class ZooBecoApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(LoadingScreen(name='loading'))
        sm.add_widget(WelcomeScreen(name='welcome'))
        sm.add_widget(ClassificationScreen(name='classification'))
        return sm

if __name__ == '__main__':
    ZooBecoApp().run()
