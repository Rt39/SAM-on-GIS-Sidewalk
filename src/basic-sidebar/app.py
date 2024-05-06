import seaborn as sns
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from shiny.types import FileInfo
from shared import predict_mask, predict_centroids, parse_centroids, predict_mask_with_centroids

from shiny.express import input, render, ui

ui.page_opts(title="Lorem Ipsum")

with ui.sidebar():
    ui.input_file("file", label="Upload file", accept=".tif", multiple=False)

    @render.plot
    def preview():
        file: list[FileInfo] | None = input.file()
        if file is None:
            return None
        img = tifffile.imread(file[0]["datapath"])
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_axis_off()
        return fig
    

with ui.layout_columns(col_widths=[4, 4, 4]):
    with ui.card():
        ui.card_header("Raw Prediction")
        @render.plot
        def raw_pred():
            file: list[FileInfo] | None = input.file()
            if file is None:
                return None
            img = tifffile.imread(file[0]["datapath"])
            masks = predict_mask(img)
            fig, ax = plt.subplots()
            ax.imshow(masks, cmap="gray")
            ax.set_axis_off()
            return fig
        
    with ui.card():
        ui.card_header("Image with Centroids")
        @render.plot
        def img_centroids():
            file: list[FileInfo] | None = input.file()
            if file is None:
                return None
            img = tifffile.imread(file[0]["datapath"])
            centroids = predict_centroids(img)
            print(centroids)
            centroids = parse_centroids(centroids)
            print(centroids)
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.scatter(centroids[:, 0], centroids[:, 1], color="red")
            ax.set_axis_off()
            return fig

        
    with ui.card():
        ui.card_header("Prediction with Centroids")
        @render.plot
        def centroids_pred():
            file: list[FileInfo] | None = input.file()
            if file is None:
                return None
            img = tifffile.imread(file[0]["datapath"])
            centroids = predict_centroids(img)
            centroids = parse_centroids(centroids)
            masks = predict_mask_with_centroids(img, centroids[:, :2])
            fig, ax = plt.subplots()
            ax.imshow(masks, cmap="gray")
            ax.scatter(centroids[:, 0], centroids[:, 1], color="red")
            ax.set_axis_off()
            return fig