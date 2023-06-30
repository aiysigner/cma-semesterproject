import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from h3 import h3
from shapely.geometry import Polygon
import folium

class RasterBasedEntity:
    def __init__(self):
        self.visits = pd.DataFrame()
        self.locations = gpd.GeoDataFrame()

    def create_locations(self):
        # Group the visits by location
        locations = self.visits.groupby('location')

        # Initialize the locations DataFrame
        self.locations = pd.DataFrame()

        # Compute the statistics for each unique location
        self.locations['num_visits'] = locations.size()
        self.locations['avg_time'] = locations['duration'].mean()
        self.locations['avg_time'] = self.locations['avg_time'].dt.round('S')
        self.locations['total_time'] = locations['duration'].sum()

        # Reset the index
        self.locations.reset_index(inplace=True)

        # Compute the polygon geometries for each hexbin
        self.locations['geometry'] = self.locations['location'].apply(lambda x: Polygon(h3.h3_to_geo_boundary(x, geo_json=True)))

        # Convert the DataFrame to a GeoDataFrame
        self.locations = gpd.GeoDataFrame(self.locations, geometry='geometry')

    def map_locations(self, column='num_visits'):
        # Initialize the map at the center of all locations
        map_center = [47.356, 8.673] # Greifensee
        
        # Create a folium Map
        m = folium.Map(location=map_center, zoom_start=11, max_zoom=13, tiles='cartodbpositron', control_scale=True)
        
        # If column is avg_time, convert it to seconds for normalization
        if column == 'avg_time' or column == 'total_time':
            column_values = self.locations[column].dt.total_seconds()
        else:
            column_values = self.locations[column]

        # Normalize the chosen column for coloring
        norm = plt.Normalize(column_values.min(), column_values.max())
        cmap = plt.get_cmap("YlOrRd")  # choose the color scheme
        rgb = [cmap(norm(value)) for value in column_values]  # map the normalized values to rgb
        hex_colors = ["#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in rgb]  # convert rgb to hex

        # Add polygons to the map
        for geometry, color in zip(self.locations.geometry, hex_colors):
            locations = [[y, x] for x, y in list(geometry.exterior.coords)]
            folium.Polygon(locations, fill_color=color, color='black', weight=1, fill_opacity=0.8).add_to(m)

        return m
        

class RasterBasedPerson(RasterBasedEntity):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.visits = pd.DataFrame(columns=['start_time', 'end_time', 'location', 'duration'])

    @classmethod
    def from_csv(cls, filename, hex_resolution, visit_threshold=pd.Timedelta('10 minute')):
        data = pd.read_csv(filename)

        # Remove rows with missing coordinates
        data = data.dropna(subset=['lat_y', 'lon_x'])

        # Convert the datetime to a pandas datetime object
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # Sort the data by datetime
        data = data.sort_values('datetime')

        # Create the hexbin column
        data['hexbin'] = data.apply(lambda row: h3.geo_to_h3(row['lat_y'], row['lon_x'], hex_resolution), axis=1)

        # Initialize the person
        person = cls(user_id=data.iloc[0]['user_id'])

        # Initialize the first visit with the first row
        current_visit = {'start_time': data.iloc[0]['datetime'], 'end_time': data.iloc[0]['datetime'], 'location': data.iloc[0]['hexbin']}

        # Iterate over the rest of the rows
        for i in range(1, len(data)):
            current_row = data.iloc[i]

            # If the hexbin has changed, it's potentially the end of a visit
            if (current_row['hexbin'] != current_visit['location']):
                # Update the end time of the last visit
                current_visit['end_time'] = data.iloc[i - 1]['datetime']

                # Compute the duration of the visit
                current_visit['duration'] = current_visit['end_time'] - current_visit['start_time']

                # Check if the visit duration is long enough to be considered a valid visit
                if current_visit['duration'] >= visit_threshold:
                    # Add the visit to the person
                    person.visits = pd.concat([person.visits, pd.DataFrame([current_visit])], ignore_index=True)

                # Start a new visit
                current_visit = {'start_time': current_row['datetime'], 'end_time': current_row['datetime'], 'location': current_row['hexbin']}

        # Don't forget to check and potentially add the last visit
        current_visit['end_time'] = data.iloc[-1]['datetime']
        current_visit['duration'] = current_visit['end_time'] - current_visit['start_time']
        if current_visit['duration'] >= visit_threshold:
            person.visits = pd.concat([person.visits, pd.DataFrame([current_visit])], ignore_index=True)

        # After all visits are processed, create the locations DataFrame
        person.create_locations()
        
        return person
    

class RasterBasedUnionGroup(RasterBasedEntity):
    def __init__(self):
        super().__init__()
        self.people = []
        self.visits = pd.DataFrame()

    def add_person(self, person):
        self.people.append(person)

    def compute_visits(self):
        # Combine all visits from all people
        self.visits = pd.concat([person.visits for person in self.people], ignore_index=True)

        # Compute the locations DataFrame
        self.create_locations()


class RasterBasedIntersectGroup(RasterBasedEntity):
    def __init__(self):
        super().__init__()
        self.people = []
        self.visits = pd.DataFrame()

    def add_person(self, person):
        self.people.append(person)

    def compute_visits(self, n_min=2, n=None, min_duration=pd.Timedelta('5 minutes')):
        # Split each visit into one-minute intervals
        intervals_df = pd.DataFrame()
        for person in self.people:
            for _, visit in person.visits.iterrows():
                time_range = pd.date_range(start=visit['start_time'], end=visit['end_time'], freq='T')
                df = pd.DataFrame({'time': time_range, 'location': visit['location']})
                intervals_df = pd.concat([intervals_df, df], ignore_index=True)

        # Remove seconds from the time values
        intervals_df['time'] = intervals_df['time'].dt.floor('T')

        # Group by location and time, and count the number of occurrences
        intervals_count = intervals_df.groupby(['location', 'time']).size().reset_index(name='counts')

        # Filter out the visits where the number of occurrences is less than n or n_min
        if n is None:
            common_intervals = intervals_count[intervals_count['counts'] >= n_min]
        else:
            common_intervals = intervals_count[intervals_count['counts'] == n]

        # Sort by time and group by location, and get the start and end times of continuous time blocks
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
