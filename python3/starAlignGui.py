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
import DataModel
from kivy.logger import Logger,LOG_LEVELS

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class MergeDialog(FloatLayout):
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
            root= App.get_running_app().root
            text=rv.data[index]["text"]
            path=rv.data[index]["path"]
            root.preview(self)

class starAlignBL(BoxLayout):

    
    def __init__(self, **kwargs):
        super(starAlignBL, self).__init__(**kwargs)
        self.data_model=DataModel.DataModel()
        self.data_model.logger=Logger
        self.img=None
        self.texture=None
        self.focal_length=0
        self.crop_factor=0
        self.img_list=None

    def show_img(self):
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

    def preview(self,instance):
        index=self.img_list.index(instance.text)
        Logger.debug(f"Previewing {instance.text}, index={index}")
        self.img=cv2.flip(self.data_model.images[index].original_image,0)
        texture_size=(self.img.shape[1],self.img.shape[0])
        self.texture = Texture.create(size=texture_size,
                                 colorfmt='rgb',
                                 bufferfmt='ushort')
        self.texture.blit_buffer(self.img.tobytes(),bufferfmt='ushort')
        self.show_img()

    def show_on_size(self):
        Logger.debug("Show on window resize")
        if self.img is not None:
            self.show_img()

    def dismiss_popup(self):
        self._popup.dismiss()
        
    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9), auto_dismiss=False)
        self._popup.open()

    def show_merge(self):
        content = MergeDialog(cancel=self.dismiss_popup)
        self._popup = Popup(title="Merging images", content=content,
                            size_hint=(0.9, 0.9), auto_dismiss=False)
        self._popup.open()
        
    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9), auto_dismiss=False)
        self._popup.open()

    def load(self, path, filename):
        if len(filename) > 0: #if filename is not empty
            #Take the file in middle as reference
            sortedImages=sorted(filename)
            ref=sortedImages[len(filename)//2]
            sortedImages.remove(ref)
            newlist=[ref]+sortedImages
            filelist=[f'{os.path.split(x)[1]}' for x in newlist]
            # Add the filenames to the ListView
            self.ids.pics_list.data=[{"text":x,"path":y} for x,y in zip(filelist,newlist)]
            self.img_list=filelist
            Logger.debug(self.ids.pics_list.data)
            self.dismiss_popup()
            root=App.get_running_app().root
            root.ids['merge_button'].disabled=False
            root.show_set_focal()
            for p in newlist:
                Logger.debug(f"Load image: {p}, focal_length={self.focal_length} crop_factor={self.crop_factor}")
                self.data_model.add_image(p,focal_length=self.focal_length,
                                          crop_factor=self.crop_factor)
        else:
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
        try:
            self.focal_length = float(focal_length)
            self.crop_factor = float(crop_factor)
        except:
            pass
        root = App.get_running_app().root
        root.ids["focus_label"].text=f"Focus Length: {self.focal_length}mm  Crop Factor: {self.crop_factor}"
        Logger.debug(f"Set Focus Length: {self.focal_length}mm")
        Logger.debug(f"Set Crop Factor: {self.crop_factor}")
        self.dismiss_popup()



class starAlignApp(App):
    def build(self):
        return starAlignBL()


if __name__ == "__main__":
    Config.set("kivy","exit_on_escape","0")
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    Logger.setLevel(LOG_LEVELS["debug"])
    starAlignApp().run()
