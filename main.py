
import cv2 as cv
from flask import Flask, render_template, request,redirect,url_for
from flask_bootstrap import Bootstrap
from sklearn.cluster import KMeans
import numpy as np
from tkinter import filedialog as fd
import tkinter as tk

n_clusters = 10
UPLOAD_FOLDER = 'static/'
image_name = "Shubham.png"

app = Flask(__name__)
app.config['SECRET_KEY'] = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Bootstrap(app)

def color_pallete(FILENAME, n_clusters=5):
    def palette(clusters):
        width = 300
        palette = np.zeros((50, width, 3), np.uint8)
        steps = width/clusters.cluster_centers_.shape[0]
        for idx, centers in enumerate(clusters.cluster_centers_):
            palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
        return palette

    img_2 = cv.imread(FILENAME)
    img_2 = cv.resize(img_2, (100, 100))
    img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)

    clt_3 = KMeans(n_clusters)
    clt_3.fit(img_2.reshape(-1, 3))

    pallete_lista = np.ndarray.tolist(palette(clt_3))

    pallete_lista_final = []
    for x in pallete_lista[0]:
        if x not in pallete_lista_final:
            pallete_lista_final.append(x)

    img_palette_hex = [
        ('#%02x%02x%02x' % tuple(color)).upper()
        for color in pallete_lista_final
    ]

    return img_palette_hex






# all Flask routes below
@app.route("/")
def home():
    global image_name
    image_name = "default.jpg"

    return render_template("index.html")


@app.route("/imgcolorpallete")
def imgcolorpallete():
    global image_name, n_clusters
    filename = image_name
    print(filename)
    if filename:
        img_palette = color_pallete(filename, n_clusters)
        file_name=filename.split("/")[-1]
        image_cv2 = cv.imread(filename)
        result=cv.imwrite(f'static/{file_name}', image_cv2)
        return render_template(
                'color_pallet.html',
                img_pallete=img_palette,
                img_url=file_name
                )
    return redirect(request.url)




@app.route('/get_image',)
def get_image():
    global image_name
    root =tk.Tk()
    filename = fd.askopenfilename(initialdir='D:\python\Day91')
    if filename:
        image_name = filename

        root.destroy()

        return redirect(url_for('imgcolorpallete'))

    root.destroy()
    return render_template("index.html")




if __name__ == '__main__':
    app.run(debug=True)