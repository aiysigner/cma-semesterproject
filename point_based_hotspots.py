import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from shapely.geometry import Point
from haversine import haversine
import folium
import folium.plugins

class PointBasedEntity:
    def __init__(self):
        self.visits = pd.DataFrame()
        self.locations = gpd.GeoDataFrame()

    def create_locations(self, cluster_radius=0.007):        
        # Convert locations to 2D numpy array
        points = np.array(self.visits['location'].tolist())
        
        # Create the distance matrix
        dist_matrix = squareform(pdist(points, metric='euclidean'))

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=cluster_radius, metric='precomputed', min_samples=1)
        labels = dbscan.fit_predict(dist_matrix)

        # Create a copy of the visits DataFrame
        visits_with_labels = self.visits.copy()
        visits_with_labels['label'] = labels

        # Group the visits by location label
        locations = visits_with_labels.groupby('label')

        # Initialize the locations DataFrame
        self.locations = pd.DataFrame()

        # Compute the statistics for each unique location
        self.locations['num_visits'] = locations.size()
        self.locations['avg_time'] = locations['duration'].mean()
        self.locations['avg_time'] = self.locations['avg_time'].dt.round('S')
        self.locations['total_time'] = locations['duration'].sum()

        # Compute the location centroid for each group
        self.locations['location'] = locations['location'].apply(lambda x: np.mean(np.array(x.tolist()), axis=0))

        # Convert the DataFrame to a GeoDataFrame
        self.locations = gpd.GeoDataFrame(self.locations, geometry=self.locations['location'].apply(lambda x: Point(x[0], x[1])))

    def map_locations(self, column='num_visits'):
        # Initialize the map at the center of all locations (of person 0)
        map_center = [47.408675, 8.833210]

        # Create a folium Map
        m = folium.Map(location=map_center, zoom_start=11, max_zoom=13, tiles='cartodbpositron', control_scale=True)

        # If column is avg_time or total_time, convert it to seconds for normalization
        if column == 'avg_time' or column == 'total_time':
            column_values = self.locations[column].dt.total_seconds()
        else:
            column_values = self.locations[column]

        # Normalize the chosen column for gradient
        norm = plt.Normalize(column_values.min(), column_values.max())

        # Calculate normalized values for each location
        normalized_values = [norm(value) for value in column_values]

        # Define the gradient
        gradient = {
            0.0: '#ffffcc',
            0.1: '#fff1a9',
            0.2: '#fee187',
            0.3: '#feca66',
            0.4: '#feab49',
            0.5: '#fd8c3c',
            0.6: '#fc5b2e',
            0.7: '#ed2e21',
            0.8: '#d41020',
            0.9: '#b00026',
            1.0: '#800026'
        }

        # Add HeatMap to the map
        data = [[row['geometry'].x, row['geometry'].y, normalized_values[idx]] for idx, row in self.locations.iterrows()]
        folium.plugins.HeatMap(data, gradient=gradient).add_to(m)

        return m

    
class PointBasedPerson(PointBasedEntity):
    def __init__(self, user_id, dist_threshold):
        super().__init__()
        self.user_id = user_id
        self.dist_threshold = dist_threshold  # Distance threshold in km
        self.visits = pd.DataFrame(columns=['start_time', 'end_time', 'location', 'duration'])

    @classmethod
    def from_csv(cls, filename, dist_threshold, visit_threshold=pd.Timedelta('10 minute')):
        data = pd.read_csv(filename)
        
        # Remove rows with missing coordinates
        data = data.dropna(subset=['lat_y', 'lon_x'])

        # Convert the datetime to a pandas datetime object
        data['datetime'] = pd.to_datetime(data['datetime'])

        # Sort the data by datetime
        data = data.sort_values('datetime')

        # Initialize the person
        person = cls(user_id=data.iloc[0]['user_id'], dist_threshold=dist_threshold)

        # Initialize the first visit with the first row
        current_visit = {
            'start_time': data.iloc[0]['datetime'],
            'end_time': data.iloc[0]['datetime'],
            'location': (data.iloc[0]['lat_y'], data.iloc[0]['lon_x']),
            'points': [(data.iloc[0]['lat_y'], data.iloc[0]['lon_x'])]
        }

        # Iterate over the rest of the rows
        for i in range(1, len(data)):
            current_row = data.iloc[i]
            current_location = (current_row['lat_y'], current_row['lon_x'])

            # Calculate centroid of current visit
            lat_centroid = sum(p[0] for p in current_visit['points']) / len(current_visit['points'])
            lon_centroid = sum(p[1] for p in current_visit['points']) / len(current_visit['points'])
            centroid = (lat_centroid, lon_centroid)

            # If the haversine distance to the centroid is above the threshold, it's the end of a visit
            if haversine(centroid, current_location) > person.dist_threshold:
                # Update the end time of the last visit
                current_visit['end_time'] = data.iloc[i - 1]['datetime']

                # Compute the duration of the visit
                current_visit['duration'] = current_visit['end_time'] - current_visit['start_time']
                
                # Update the location of the visit
                current_visit['location'] = centroid

                # Check if the visit duration is long enough to be considered a valid visit
                if current_visit['duration'] >= visit_threshold:
                    # Add the visit to the person
                    person.visits = pd.concat([person.visits, pd.DataFrame([current_visit])], ignore_index=True)

                # Start a new visit
                current_visit = {
                    'start_time': current_row['datetime'],
                    'end_time': current_row['datetime'],
                    'location': current_location,
                    'points': [current_location]
                }
            else:
                # Add the current location to the points of the current visit
                current_visit['points'].append(current_location)

        # Don't forget to check and potentially add the last visit
        current_visit['end_time'] = data.iloc[-1]['datetime']
        current_visit['duration'] = current_visit['end_time'] - current_visit['start_time']
        if current_visit['duration'] >= visit_threshold:
            person.visits = pd.concat([person.visits, pd.DataFrame([current_visit])], ignore_index=True)

        # After all visits are processed, create the locations DataFrame
        person.create_locations()

        return person

    
class PointBasedUnionGroup(PointBasedEntity):
    def __init__(self):
        super().__init__()
        self.people = []
        self.visits = pd.DataFrame()

    def add_person(self, person):
        assert isinstance(person, PointBasedPerson), "Input person is not an instance of the PointBasedPerson class."
        self.people.append(person)

    def compute_visits(self):
        # Combine all visits from all people
        self.visits = pd.concat([person.visits for person in self.people], ignore_index=True)

        # Compute the locations DataFrame
        self.create_locations()

        
class PointBasedIntersectGroup(PointBasedEntity):
    def __init__(self):
        super().__init__()
        self.people = []
        self.visits = pd.DataFrame()

    def add_person(self, person):
        self.people.append(person)

    def compute_visits(self, n_min=2, n=None, min_duration=pd.Timedelta('5 minutes'), cluster_radius=0.007):
        # Concatenate all visits from all people
        all_visits = pd.concat([person.visits for person in self.people], ignore_index=True)
        
        # Create separate 'lat' and 'lon' columns
        all_visits[['lat', 'lon']] = pd.DataFrame(all_visits['location'].tolist(), index=all_visits.index)

        # Compute the distance matrix
        dist_matrix = squareform(pdist(all_visits[['lat', 'lon']], metric='euclidean'))

        # Apply DBSCAN on the distance matrix
        dbscan = DBSCAN(eps=cluster_radius, metric='precomputed', min_samples=1)
        labels = dbscan.fit_predict(dist_matrix)

        # Assign cluster labels to the visits
        all_visits['cluster'] = labels

        # Compute the centroid for each cluster and assign to each visit within the cluster
        centroid_df = all_visits.groupby('cluster')[['lat', 'lon']].mean().rename(columns={'lat': 'centroid_lat', 'lon': 'centroid_lon'})
        all_visits = all_visits.join(centroid_df, on='cluster')
        all_visits['location'] = list(zip(all_visits['centroid_lat'], all_visits['centroid_lon']))

        # Split each visit into one-minute intervals and assign the cluster label to the segments
        intervals_df = pd.DataFrame()
        for _, visit in all_visits.iterrows():
            time_range = pd.date_range(start=visit['start_time'], end=visit['end_time'], freq='T')
            df = pd.DataFrame({'time': time_range, 'location': [visit['location']]*len(time_range)})
            intervals_df = pd.concat([intervals_df, df], ignore_index=True)

        # Remove seconds from the time values
        intervals_df['time'] = intervals_df['time'].dt.floor('T')

        # Group by cluster and time, and count the number of occurrences
        intervals_count = intervals_df.groupby(['location', 'time']).size().reset_index(name='counts')

        # Filter out the visits where the number of occurrences is less than n or n_min
        if n is None:
            common_intervals = intervals_count[intervals_count['counts'] >= n_min]
        else:
            common_intervals = intervals_count[intervals_count['counts'] == n]

        # Sort by time and group by cluster, and get the start and end times of continuous time blocks
        common_intervals = common_intervals.sort_values('time')
        common_intervals['block'] = (common_intervals['time'].diff() > pd.Timedelta('1 minute')).cumsum()
        common_visits = common_intervals.groupby(['location', 'block']).agg({'time': ['min', 'max'], 'counts': 'size'}).reset_index()
        common_visits.columns = ['location', 'block', 'start_time', 'end_time', 'counts']

        # Convert the counts of times to durations
        common_visits['duration'] = common_visits['counts'] * pd.Timedelta('1 minutes')

        # Filter out the visits where the duration is less than the minimum duration
        common_visits = common_visits[common_visits['duration'] >= min_duration]

        # Drop the unnecessary columns
        common_visits = common_visits.drop(columns=['block', 'counts'])
        
        # Update the visits attribute
        self.visits = common_visits

        # Compute the locations DataFrame
        self.create_locations()
