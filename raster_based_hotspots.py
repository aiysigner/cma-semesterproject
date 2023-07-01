import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from h3 import h3
from shapely.geometry import Polygon
import folium

class RasterBasedEntity:
    """
    A base class for raster-based entities.

    This class is used to implement common functionalities shared between raster-based
    entities such as people and groups of people.
    
    Attributes:
        visits (pd.DataFrame): A DataFrame to hold the visit data.
        locations (gpd.GeoDataFrame): A GeoDataFrame to hold the location data.
    """

    def __init__(self):
        """
        Initializes the RasterBasedEntity with empty visits and locations dataframes.
        """

        self.visits = pd.DataFrame()
        self.locations = gpd.GeoDataFrame()

    def create_locations(self):
        """
        Creates the locations dataframe.

        This method groups the visits by location and computes several statistics for each 
        location such as the number of visits, average visit duration, and total visit duration. 
        It also computes the polygon geometries for each location based on the hexbin IDs using the H3 library.
        """

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
        """
        Creates a folium Map object with locations visualized as polygons of hexagons.

        This method creates a folium Map object and adds a Polygon layer to it for each location, 
        where the fill color of the polygon corresponds to the value of the specified column for that location.

        Args:
            column (str, optional): The name of the column in the locations dataframe to use for coloring the polygons. 
                                    Defaults to 'num_visits'.

        Returns:
            folium.Map: The created folium Map object.
        """

        # Initialize the map at the center of the Canton of Zürich
        map_center = [47.412724, 8.655083]
        
        # Create a folium Map
        m = folium.Map(location=map_center, zoom_start=10, max_zoom=13, tiles='cartodbpositron', control_scale=True)
        
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
    """
    A class that represents a raster-based person.

    This class is used to represent a person in a raster-based framework. It inherits from the
    RasterBasedEntity base class.

    Attributes:
        user_id (str): The ID of the person.
        visits (pd.DataFrame): A DataFrame to hold the visit data.
        locations (gpd.GeoDataFrame): A GeoDataFrame to hold the location data.
    """

    def __init__(self, user_id):
        """
        Initializes the RasterBasedPerson with the specified user ID and an empty visits dataframe.

        Args:
            user_id (str): The ID of the person.
        """

        super().__init__()
        self.user_id = user_id
        self.visits = pd.DataFrame(columns=['start_time', 'end_time', 'location', 'duration'])

    @classmethod
    def from_csv(cls, filename, hex_resolution, visit_threshold=pd.Timedelta('10 minute')):
        """
        Creates a RasterBasedPerson object from a CSV file.

        This method reads geotracking data from a CSV file, identifies separate visits, and creates a RasterBasedPerson object.

        Args:
            filename (str): The name of the CSV file to read the data from.
            hex_resolution (int): The H3 hexagon resolution to use for rasterizing the locations.
            visit_threshold (pd.Timedelta, optional): The minimum duration of a visit to be considered valid. 
                                                      Defaults to pd.Timedelta('10 minute').

        Returns:
            RasterBasedPerson: A RasterBasedPerson object with the identified visits.
        """

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
    """
    A class representing a group of raster-based people.

    This class is used to represent a group of people in a raster-based framework. The group is created 
    through the union of individuals, aggregating the visit data of all its members. It inherits from the 
    RasterBasedEntity base class.

    Attributes:
        people (list): A list of RasterBasedPerson objects.
        visits (pd.DataFrame): A DataFrame to hold the visit data.
        locations (gpd.GeoDataFrame): A GeoDataFrame to hold the location data.
    """

    def __init__(self):
        """
        Initializes the RasterBasedUnionGroup with an empty list of people and an empty visits dataframe.
        """

        super().__init__()
        self.people = []
        self.visits = pd.DataFrame()

    def add_person(self, person):
        """
        Adds a person to the group.

        This method adds a RasterBasedPerson object to the group's list of people.

        Args:
            person (RasterBasedPerson): The person to be added.

        Raises:
            AssertionError: If the input person is not an instance of the RasterBasedPerson class.
        """

        assert isinstance(person, RasterBasedPerson), "Input person is not an instance of the RasterBasedPerson class."
        self.people.append(person)

    def compute_visits(self):
        """
        Computes the visit data for the group.

        This method concatenates the visit data of all people in the group and computes the locations dataframe.
        """

        # Combine all visits from all people
        self.visits = pd.concat([person.visits for person in self.people], ignore_index=True)

        # Compute the locations DataFrame
        self.create_locations()


class RasterBasedIntersectGroup(RasterBasedEntity):
    """
    A class representing a group of raster-based people intersected by shared visits.

    This class is used to represent a group of people in a raster-based framework where the group's visits 
    represent the intersection of visits made by all group members. It inherits from the RasterBasedEntity base class.

    Attributes:
        people (list): A list of RasterBasedPerson objects.
        visits (pd.DataFrame): A DataFrame to hold the visit data.
        locations (gpd.GeoDataFrame): A GeoDataFrame to hold the location data.
    """

    def __init__(self):
        """
        Initializes the RasterBasedIntersectGroup with an empty list of people and an empty visits dataframe.
        """

        super().__init__()
        self.people = []
        self.visits = pd.DataFrame()

    def add_person(self, person):
        """
        Adds a person to the group.

        This method adds a RasterBasedPerson object to the group's list of people.

        Args:
            person (RasterBasedPerson): The person to be added.

        Raises:
            AssertionError: If the input person is not an instance of the RasterBasedPerson class.
        """

        assert isinstance(person, RasterBasedPerson), "Input person is not an instance of the RasterBasedPerson class."
        self.people.append(person)

    def compute_visits(self, n_min=2, n=None, min_duration=pd.Timedelta('5 minutes')):
        """
        Computes the intersected visit data for the group.

        This method computes the intersected visit data for the group, which represent the visits made by all group 
        members that are common among at least `n_min` number of members and that last for at least `min_duration`.

        Args:
            n_min (int): The minimum number of members that need to have made a visit for it to be considered common.
            n (int, optional): The exact number of members that need to have made a visit for it to be considered common. 
                               If not provided, any visit made by at least `n_min` number of members is considered common.
            min_duration (pd.Timedelta): The minimum duration a visit must last for it to be considered valid.
        """

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
