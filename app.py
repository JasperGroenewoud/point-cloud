from pathlib import Path
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from munch import Munch

from viktor import ViktorController, progress_message, File
from viktor.views import ImageView, ImageResult, DataGroup, DataItem, DataResult, DataView
from viktor.parametrization import ViktorParametrization, TextField, NumberField, Tab, Step, FileField, OptionField, \
    LineBreak, Text, BooleanField, IsFalse, Lookup


def get_input_headers(params, **kwargs):
    #xyz_file: File = params.step_1.input_1.file
    xyz_file: File = use_correct_file(params)

    # Get headers
    with xyz_file.open_binary() as f:
        df = pd.read_csv(f, delimiter=';')
        headers = df.columns.tolist()
        headers = [item for item in headers if item not in ['X', 'Y', 'Z']]
    return headers
def use_correct_file(params: Munch):
    if params.step_1.input_7 is True:
        params.xyz_file = File.from_path(
            Path(__file__).parent / "37GN1_21 - Haringvliet Bridge.xyz"
        )
    else:
        params.xyz_file = params.step_1.input_1.file
    return params.xyz_file

class Parametrization(ViktorParametrization):
    step_1 = Step('Step 1: Explore', views=['create_point_cloud_non_filter_3d', 'create_point_cloud_quick_selection'])
    step_1.not_in_params = Text(
        '### 3D Point Cloud Clustering with K-Means and Python\n'
        '**Use this app as a tool to explore your 3D (LiDAR/ Point Cloud) datasets. We have provided an example of a highway segment with two gantries.'
        'The app and code have been inspired [by this article by Florent Poux.](https://medium.com/towards-data-science/3d-point-cloud-clustering-tutorial-with-k-means-and-python-c870089f3af8)**\n'
        '\n'
        'The app is structured into 3 steps:\n'
        '- **Step 1:** upload the 3D data in .xyz format, and explore the data using 3D and 2D plots. \n'
        '- **Step 2:** filter the data,if applicable. The filter available in this app filters out pionts with a value below the mean, for the user-specified axis (X, Y or Z). \n'
        '- **Step 3:** cluster the data to identify objects. \n'
        '#### Step 1: Explore \n'
        '1. Upload an .xyz file below. Ensure that all columns are named. The first three columns must be the X, Y and Z values, and be named exactly X, Y and Z.\n'
        '2. The 3D representation will be generated in the "3D visualization" tab \n'
        '3. Next, select the feature you would like to plot, such as R, G, B or Intensity. \n'
        '4. Then, experiment with the scatter plots in the "2D representation" tab. The parameters enable you to change the axes, and the value plotted as an axis average. Use this to identify along which axis to filer. Notice how the ground and water surface all fall below the Z-axis mean, so we can use that as a filter in the next step. \n'
        '\n'
        '#### Example: highway with gantries \n'
        'We have provided a sample file to cluster a highway with two gantries, which is partly on land and partly over water. The filter is along the Z-axis, to eliminate the water and ground, and keep only the road surface.')
    step_1.input_7 = BooleanField('Use sample highway file', default=True)
    step_1.input_1 = FileField('Upload an .xyz file', file_types=['.xyz'], max_size=90_000_000, visible=IsFalse(Lookup("step_1.input_7")))
    step_1.lb_1 = LineBreak()
    step_1.option_3 = OptionField('Plotted feature', options=get_input_headers, default="Classification")
    step_1.lb_4 = LineBreak()
    step_1.option_1 = OptionField('2D scatter plot 1 | x-axis', options=['X', 'Y', 'Z'], default='X')
    step_1.option_2 = OptionField('2D scatter plot 1 | y-axis', options=['X', 'Y', 'Z'], default='Z')
    step_1.lb_2 = LineBreak()
    step_1.option_5 = OptionField('2D scatter plot 2 | x-axis', options=['X', 'Y', 'Z'], default='Y')
    step_1.option_6 = OptionField('2D scatter plot 2 | y-axis', options=['X', 'Y', 'Z'], default='Z')
    step_1.lb_3 = LineBreak()
    step_1.option_4 = OptionField('2D Horizontal average line variable', options=['X', 'Y', 'Z'], default='Z')
    step_2 = Step('Step 2: Filter', views=['create_point_cloud_filter_3d', 'create_point_cloud_filter_2d'])
    step_2.not_in_params = Text(
        '#### Step 2: Filter \n'
        'This step enables you to filter out data, like the water and ground surface below the highway. \n'
        '1. Based on your conclusion from the previous step, select the axis you want to use as filter. In the highway sample case, this would be the Z-axis. \n'
        '2. Also select the data feature for the plot. Feel free to experiment with the different features for the highway case. \n'
    )
    step_2.option_1 = OptionField('Axis as filter:', options=['X','Y','Z'], default='Z')
    step_2.lb_1 = LineBreak()
    step_2.option_2 = OptionField('Feature for plot:', options=get_input_headers, default="Classification")
    step_3 = Step('Step 3: Cluster', views=['spatial_clustering_3d', 'spatial_clustering', 'elbow_method'])
    step_3.not_in_params = Text(
        '#### Step 3: Cluster \n'
        'This step enables you to cluster the data. \n'
        '1. In the highway example, set the number of clusters to 3. The output will distinguish between the two highway segments with gantries, and the one without.  \n'
        '2. The "Elbow" tab helps identify the ideal number for other datasets: the "elbow" point suggests a likely suitable number of clusters.'
    )
    step_3.input_1 = NumberField('Number of clusters', default=3, min=1, max=30, step=1)


class Controller(ViktorController):
    label = "Point Cloud quick selection"
    parametrization = Parametrization

    def get_headers(self, params, **kwargs):
        #xyz_file: File = params.step_1.input_1.file
        xyz_file: File = use_correct_file(params)
        # Get headers
        with xyz_file.open_binary() as f:
            df = pd.read_csv(f, delimiter=';')
        return df

    @ImageView("3D visualization", duration_guess=10)
    def create_point_cloud_non_filter_3d(self, params, **kwargs):
        df = self.get_headers(params)

        fig=plt.figure()

        ax = plt.axes(projection='3d')
        if params.step_1.option_3 is not None:
            ax.scatter(df['X'], df['Y'], df['Z'], c = df[params.step_1.option_3], s=0.1)
        else:
            ax.scatter(df['X'], df['Y'], df['Z'], s=0.1)

        png_data = BytesIO()
        fig.savefig(png_data, format='png')

        return ImageResult(png_data)


    @ImageView("2D representation", duration_guess=10)
    def create_point_cloud_quick_selection(self, params, **kwargs):
        df = self.get_headers(params)
        progress_message("Plot results")
        fig = plt.figure()

        plt.subplot(1, 2, 1)  # row 1, col 2 index 1
        if params.step_1.option_3 is not None:
            plt.scatter(df[params.step_1.option_1], df[params.step_1.option_2], c=df[params.step_1.option_3], s=0.05)
        else:
            plt.scatter(df[params.step_1.option_1], df[params.step_1.option_2], s=0.05)

        plt.axhline(y=np.mean(df[params.step_1.option_4]), color='r', linestyle='-'),
        plt.title("First view"),
        plt.xlabel(params.step_1.option_1),
        plt.ylabel(params.step_1.option_2)

        plt.subplot(1, 2, 2)  # index 2
        if params.step_1.option_3 is not None:
            plt.scatter(df[params.step_1.option_5], df[params.step_1.option_6], c=df[params.step_1.option_3], s=0.05)
        else:
            plt.scatter(df[params.step_1.option_5], df[params.step_1.option_6], s=0.05)

        plt.axhline(y=np.mean(df[params.step_1.option_4]), color='r', linestyle='-')
        plt.title("Second view")
        plt.xlabel(params.step_1.option_5)
        plt.ylabel(params.step_1.option_6)

        plt.plot()

        png_data = BytesIO()
        fig.savefig(png_data, format='png')
        plt.close()

        return ImageResult(png_data)

    @ImageView("3D visualization", duration_guess=10)
    def create_point_cloud_filter_3d(self, params, **kwargs):
        df = self.get_headers(params)

        fig=plt.figure()

        mask = df[params.step_2.option_1]>np.mean(df[params.step_2.option_1])

        ax = plt.axes(projection='3d')

        if params.step_2.option_2 is not None:
            ax.scatter(df['X'][mask], df['Y'][mask], df['Z'][mask], c = df[params.step_2.option_2][mask], s=0.1)
        else:
            ax.scatter(df['X'][mask], df['Y'][mask], df['Z'][mask], s=0.1)

        png_data = BytesIO()
        fig.savefig(png_data, format='png')

        return ImageResult(png_data)

    @ImageView("2D visualization", duration_guess=10)
    def create_point_cloud_filter_2d(self, params, **kwargs):
        df = self.get_headers(params)
        fig=plt.figure()

        mask = df[params.step_2.option_1]>np.mean(df[params.step_2.option_1])

        if params.step_2.option_2 is not None:
            plt.scatter(df['X'][mask], df['Y'][mask], c=df[params.step_2.option_2][mask], s=0.1)
        else:
            plt.scatter(df['X'][mask], df['Y'][mask], s=0.1)

        png_data = BytesIO()
        fig.savefig(png_data, format='png')

        return ImageResult(png_data)

    @ImageView("Clustering 3D", duration_guess=10)
    def spatial_clustering_3d(self, params, **kwargs):
        df = self.get_headers(params)
        fig=plt.figure()
        mask = df[params.step_2.option_1]>np.mean(df[params.step_2.option_1])
        ax = plt.axes(projection='3d')

        X=np.column_stack((df['X'][mask], df['Y'][mask], df['Z'][mask]))
        kmeans = KMeans(n_clusters=params.step_3.input_1).fit(X)
        ax.scatter(df['X'][mask], df['Y'][mask], df['Z'][mask], c=kmeans.labels_, s=0.1)

        png_data = BytesIO()
        fig.savefig(png_data, format='png')

        return ImageResult(png_data)

    @ImageView("Clustering 2D", duration_guess=10)
    def spatial_clustering(self, params, **kwargs):
        df = self.get_headers(params)
        fig=plt.figure()
        mask = df[params.step_2.option_1]>np.mean(df[params.step_2.option_1])
        X=np.column_stack((df['X'][mask], df['Y'][mask]))
        kmeans = KMeans(n_clusters=params.step_3.input_1).fit(X)
        plt.scatter(df['X'][mask], df['Y'][mask], c=kmeans.labels_, s=0.1)

        png_data = BytesIO()
        fig.savefig(png_data, format='png')

        return ImageResult(png_data)

    @ImageView("Elbow", duration_guess=10)
    def elbow_method(self, params, **kwargs):
        df = self.get_headers(params)
        fig=plt.figure()
        mask = df[params.step_2.option_1]>np.mean(df[params.step_2.option_1])

        X=np.column_stack((df['X'][mask], df['Y'][mask]))
        wcss = []
        for i in range(1,20):
            kmeans=KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        plt.plot(range(1,20), wcss)
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')

        png_data=BytesIO()
        fig.savefig(png_data, format='png')

        return ImageResult(png_data)