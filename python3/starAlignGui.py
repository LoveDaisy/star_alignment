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
from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
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
import threading
import cv2
import gc
import signal

def keyboardInterruptHandler(signal, frame):
    print("Thanks for using star align.".format(signal))
    exit(0)

signal.signal(signal.SIGINT, keyboardInterruptHandler)

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
    merge_text = StringProperty()

    def reset_data(self):
        self.data_model=DataModel.DataModel()
        self.data_model.logger=Logger
        self.img=None
        self.texture=None
        self.texture_stars=None
        self.focal_length=0
        self.crop_factor=0
        self.img_list=None
        self.output_name="aligned.tif"
        gc.collect()
    
    def __init__(self, **kwargs):
        super(starAlignBL, self).__init__(**kwargs)
        self.reset_data()

    def show_img(self):
        render_texture=self.ids['render_area']
        render_stars_texture=self.ids['detected_stars']
        prev_canvas=render_texture.canvas
        prev_stars_canvas=render_stars_texture.canvas
        
        prev_canvas.clear()
        prev_stars_canvas.clear()
        
        img_ratio=self.img.shape[1]/self.img.shape[0]
        render_ratio=render_texture.size[0]/render_texture.size[1]
        if img_ratio > render_ratio:
            render_size=(render_texture.size[1]*img_ratio,render_texture.size[1])
        else:
            render_size=(render_texture.size[0],render_texture.size[0]/img_ratio)
          
        render_texture.size=render_size
        with render_texture.canvas:
            Rectangle(texture=self.texture,pos=render_texture.pos, size=render_size)
        with render_stars_texture.canvas:
            Rectangle(texture=self.texture_stars,pos=render_stars_texture.pos, size=render_size)
            
    def show_blank(self):
        render_texture=self.ids['render_area']
        render_stars_texture=self.ids['detected_stars']
        prev_canvas=render_texture.canvas
        prev_stars_canvas=render_stars_texture.canvas
        prev_canvas.clear()
        prev_stars_canvas.clear()
        prev_canvas.add(Color(0.1, 0.1, 0.1, 1))
        prev_stars_canvas.add(Color(0.1, 0.1, 0.1, 1))
        rect = Rectangle(size=render_texture.size, pos=render_texture.pos)
        prev_canvas.add(rect)
        prev_stars_canvas.add(rect)

    def preview(self,instance):
        try: #not sure why selection is applied after reset. so workaround
            if instance.text == "result":
                Logger.debug(f"Previewing result")
                self.img=cv2.flip(self.result_img,0)
            else:
                index=self.img_list.index(instance.text)
                Logger.debug(f"Previewing {instance.text}, index={index}")
                self.img=cv2.flip(self.data_model.images[index].original_image,0)
                ##Create the stars based on points with opencv, therefore set dtype=uint8
                self.img_stars=np.zeros(self.img.shape,dtype=np.uint8)
                # Flip upside down
                for x,y in self.data_model.images[index].features["pts"]:
                    cv2.circle(self.img_stars, (int(x),self.img.shape[0]-int(y)), 5, (255,255,255),-1)

            # texture shape is (width, height), while cv, tiff is (height,width)
            texture_size=(self.img.shape[1],self.img.shape[0])
            self.texture = Texture.create(size=texture_size,
                                     colorfmt='rgb',
                                     bufferfmt='ushort')
            
            self.texture_stars = Texture.create(size=texture_size,
                         colorfmt='bgr',
                         bufferfmt='ubyte')
            self.texture.blit_buffer(self.img.tobytes(),bufferfmt='ushort')
            if instance.text == "result":
                self.texture_stars = self.texture
            else:
                self.texture_stars.blit_buffer(self.img_stars.tobytes(),bufferfmt='ubyte')
            self.show_img()
        except Exception as e:
            Logger.debug(f"Preview erro: {e}")

    def show_on_size(self):
        Logger.debug("Show on window resize")
        if self.img is not None:
            self.show_img()
        else:
            self.show_blank()

    def reset_ui(self):
        self.ids.pics_list.data=[]
        self.reset_data()
        self.show_blank()
        root=App.get_running_app().root
        root.ids['load_button'].disabled=False
        root.ids['merge_button'].disabled=True
        root.ids['focus_button'].disabled=True
        root.ids['reset_button'].disabled=True
        root.ids['save_button'].disabled=True
        self.ids["keep_interim"].active=False
        self.ids["detect_slider"].value=2200

    def dismiss_popup(self):
        self._popup.dismiss()
        
    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9), auto_dismiss=False)
        self._popup.open()

    def load_thread(self,root,path_list):
        for p in path_list:
            Logger.debug(f"Load image: {p}, focal_length={self.focal_length} crop_factor={self.crop_factor}")
            self.data_model.add_image(p,focal_length=self.focal_length,
                                      crop_factor=self.crop_factor)

        root.ids['load_button'].disabled=True
        root.ids['merge_button'].disabled=False
        root.ids['reset_button'].disabled=False

    def load(self, path, filename):
        if len(filename) > 0: #if filename is not empty
            #Take the file in middle as reference
            sortedImages=sorted(filename)
            ref=sortedImages[len(filename)//2]
            sortedImages.remove(ref)
            newlist=[ref]+sortedImages
            filelist=[f'{os.path.split(x)[1]}' for x in newlist]
            # Add the filenames to the ListView
            self.ids.pics_list.data=[{"text":x,"path":y,"selected":False} for x,y in zip(filelist,newlist)]
            self.img_list=filelist
            Logger.debug(self.ids.pics_list.data)
            self.dismiss_popup()
            root=App.get_running_app().root
            root.ids['focus_button'].disabled=False
            root.show_set_focal()
            threading.Thread(target=self.load_thread,args=(root,newlist)).start()
        else:
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
        for img in self.data_model.images:
            img_shape = img.fullsize_gray_image.shape
            img_size = np.array([img_shape[1], img_shape[0]])
            resize_len=int(root.ids["detect_slider"].value)
            pts, vol = DataModel.ImageProcessing.detect_star_points(img.fullsize_gray_image,resize_length=resize_len)
            sph = DataModel.ImageProcessing.convert_to_spherical_coord(pts, img_size, self.focal_length, self.crop_factor)
            feature = DataModel.ImageProcessing.extract_point_features(sph, vol)
            img.features["pts"] = pts
            img.features["sph"] = sph
            img.features["vol"] = vol
            img.features["feature"] = feature
        root = App.get_running_app().root
        root.ids["focus_label"].text=f"Focus Length: {self.focal_length}mm  Crop Factor: {self.crop_factor}"
        Logger.debug(f"Set Focus Length: {self.focal_length}mm")
        Logger.debug(f"Set Crop Factor: {self.crop_factor}")
        self.dismiss_popup()
        
    def merge_thread(self):
        self._popup.content.ok_button.disabled=True           
        keepInterim=self.ids["keep_interim"].active
        Logger.debug(f"Keep interim files: {keepInterim}")
        if self.data_model.has_image():
            self._popup.content.merge_label.text+=f"\nProcessing {self.img_list[0]}..."
            ref_img = self.data_model.images[0]

            img_shape = ref_img.fullsize_gray_image.shape
            img_size = np.array([img_shape[1], img_shape[0]])
            self.data_model.reset_result()

            self.data_model.accumulate_final_sky( np.copy(ref_img.original_image).astype("float32") / np.iinfo(
                ref_img.original_image.dtype).max)
            self._popup.content.merge_label.text+=f"\nDone"

            serial=0
            if keepInterim:
                result_img = (self.data_model.final_sky_img * np.iinfo("uint16").max).astype("uint16")
                DataModel.ImageProcessing.save_tif_image("interim00.tif", result_img, self.data_model.images[0].exif_info)

            for i in range(1,len(self.data_model.images)):
                self._popup.content.merge_label.text+=f"\nProcessing {self.img_list[i]}..."
                img=self.data_model.images[i]

                try:
                    init_pair_idx = DataModel.ImageProcessing.find_initial_match(img.features, ref_img.features)
                    tf, pair_idx = DataModel.ImageProcessing.fine_tune_transform(img.features, ref_img.features, init_pair_idx)
                    img_tf = cv2.warpPerspective(img.original_image, tf[0], tuple(img_size))
                    img_tf = img_tf.astype("float32") / np.iinfo(img_tf.dtype).max
                    self.data_model.accumulate_final_sky(img_tf)
                    serial+=1
                    if keepInterim:
                        result_img = (img_tf * np.iinfo("uint16").max).astype("uint16")
                        DataModel.ImageProcessing.save_tif_image("interim{:02d}.tif".format(serial), result_img, self.data_model.images[0].exif_info)
                    self._popup.content.merge_label.text+=f"\nDone"
                except ValueError as e:
                    Logger.debug("Alignment failed for this picture: {}. Discarded.".format(str(e)))
                    self._popup.content.merge_label.text+=f"\nCould not align, discarded!"
            self.data_model.update_final_sky()
            self.result_img = (self.data_model.final_sky_img * np.iinfo("uint16").max).astype("uint16")

            self.ids.pics_list.data.append({"text":"result","path":""})
            self.img_list.append("result")
        else:
            Logger.debug("No image to process, exiting.")
        self._popup.content.merge_label.text+=f"\nAll Done!"
        self._popup.content.ok_button.disabled=False           
        Logger.debug(f"Enable OK button {self._popup.content.ok_button.disabled}")    
        App.get_running_app().root.ids["save_button"].disabled=False
        App.get_running_app().root.ids["merge_button"].disabled=True
        Logger.debug(f"Enable Save button")
        
    def show_merge(self):
        content = MergeDialog(cancel=self.dismiss_popup)
        self._popup = Popup(title="Merging images", content=content,
                            size_hint=(0.9, 0.9), auto_dismiss=False)
        self._popup.content.ok_button.disabled=False                
        self._popup.open()
        threading.Thread(target=self.merge_thread).start() 

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9), auto_dismiss=False)
        self._popup.open()

    def save(self, path, filename):
        _, ext = os.path.splitext(filename)
        if ext.lower() not in (".tiff", ".tif"):
            filename+=".tif"
        save_path=os.path.join(path, filename)
        DataModel.ImageProcessing.save_tif_image(save_path, self.result_img, self.data_model.images[0].exif_info)
        self.dismiss_popup()
        Logger.debug(f"Result saved to {save_path}")


class starAlignApp(App):
    def build(self):
        return starAlignBL()


if __name__ == "__main__":
    Config.set("kivy","exit_on_escape","0")
    Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
    Logger.setLevel(LOG_LEVELS["debug"])
    starAlignApp().run()
