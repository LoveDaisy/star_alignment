from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.config import Config
from kivy.factory import Factory
from kivy.graphics.texture import Texture
from kivy.graphics import Color,Rectangle
from kivy.uix.popup import Popup
from kivy.uix.recycleview import RecycleView
from kivy.properties import BooleanProperty, ObjectProperty
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.label import Label

from kivy.uix.button import Button
import os

import numpy as np
import cv2
import tifffile as tiff


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SetFocusDialog(FloatLayout):
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)
    focal = ObjectProperty(None)
    crop = ObjectProperty(None)

class PicButton(Button):
    def on_release(self, **kwargs):
        super().on_release(**kwargs)
        app= App.get_running_app()
        app.root.print_widget(self)

class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    ''' Adds selection and focus behaviour to the view. '''
    
class SelectableLabel(RecycleDataViewBehavior, Label):
    ''' Add selection support to the Label '''
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        ''' Catch and handle the view changes '''
        self.index = index
        return super(SelectableLabel, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        ''' Add selection on touch down '''
        if super(SelectableLabel, self).on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        ''' Respond to the selection of items in the view. '''
        self.selected = is_selected
        if is_selected:
            app= App.get_running_app()
            text=rv.data[index]["text"]
            path=rv.data[index]["path"]
            app.root.print_widget(self)
##            print(f"selection changed to {text}, path: {path}")
##        else:
##            print("selection removed for {0}".format(rv.data[index]))

class starAlignBL(BoxLayout):
    img=None
    texture=None
    focal_length=0
    crop_factor=0
    
    def test_draw(self):
        self.img=cv2.flip(tiff.imread("DSC02690.tif"),0)
##        img=(img.astype("float32") / np.iinfo(img.dtype).max * 256).astype("uint8")
##        img=img.astype("float32") / np.iinfo(img.dtype).max
##        img=cv2.imread("DSC02690.tif")
        self.texture = Texture.create(size=(self.img.shape[1],self.img.shape[0]),
                                 colorfmt='rgb',
                                 bufferfmt='ushort')
        self.texture.blit_buffer(self.img.tobytes(),bufferfmt='ushort')
        render_texture=self.ids['render_area']

        img_ratio=self.img.shape[1]/self.img.shape[0]
        render_ratio=render_texture.size[0]/render_texture.size[1]
        if img_ratio > render_ratio:
            render_size=(render_texture.size[1]*img_ratio,render_texture.size[1])
        else:
            render_size=(render_texture.size[0],render_texture.size[0]/img_ratio)
          
        render_texture.size=render_size
        with render_texture.canvas:
            Rectangle(texture=self.texture,pos=render_texture.pos, size=render_size)

    def show_on_size(self):
        if self.img is not None:
            render_texture=self.ids['render_area']

            img_ratio=self.img.shape[1]/self.img.shape[0]
            render_ratio=render_texture.size[0]/render_texture.size[1]
            if img_ratio > render_ratio:
                render_size=(render_texture.size[1]*img_ratio,render_texture.size[1])
            else:
                render_size=(render_texture.size[0],render_texture.size[0]/img_ratio)
              
            render_texture.size=render_size
            with render_texture.canvas:
                Rectangle(texture=self.texture,pos=render_texture.pos, size=render_size)

    def print_widget(self,instance):
        print(instance.text)

    def dismiss_popup(self):
        self._popup.dismiss()
        
    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9), auto_dismiss=False)
        self._popup.open()
        
    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9), auto_dismiss=False)
        self._popup.open()

    def load(self, path, filename):
        filelist=[f'{os.path.split(x)[1]}' for x in filename]
        # Add the filenames to the ListView
        self.ids.pics_list.data=[{"text":x,"path":y} for x,y in zip(filelist,filename)]
        self.dismiss_popup()

    def save(self, path, filename):
        with open(os.path.join(path, filename), 'w') as stream:
            stream.write(self.text_input.text)

        self.dismiss_popup()

    def show_set_focal(self):
        content = SetFocusDialog(save=self.setFocal, cancel=self.dismiss_popup)
        self._popup = Popup(title="Set Focal Length", content=content,
                            size_hint=(None, None), width="300sp", height="160sp", auto_dismiss=False)
        self._popup.open()
        
    def setFocal(self,focal_length,crop_factor):
        self.focal_length = float(focal_length)
        self.crop_factor = float(crop_factor)
        print(f"Setting focal length to {self.focal_length}, crop factor to {self.crop_factor}")
        self.dismiss_popup()

##
##    def debug(self, msg):
##        pass
##
##    def warning(self, msg):
##        pass
##
##    def error(self, msg):
##        '''
##        error(self,msg)
##            A logging function for youtube_dl, will display the error status
##            Argument: msg passed in by youtube-dl
##        '''
##        pass
##
##    #@mainthread
##    def prog_hook(self,d):
##        pass
##
##    def merge(self, *args):
##        pass

class starAlignApp(App):
    def build(self):
        return starAlignBL()


if __name__ == "__main__":
    Config.set("kivy","exit_on_escape","0")
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    starAlignApp().run()
