# Proposal for Semester Project

**Patterns & Trends in Environmental Data / Computational Movement Analysis Geo 880**

| Semester:      | FS23                            |
|:---------------|:--------------------------------|
| **Data:**      | Posmo tracking data from a group of students  |
| **Title:**     | Social Hotspot Analysis         |
| **Student 1:** | Diego Gomes                     |
| **Student 2:** | Aiyana Signer                   |

## Abstract

In our project, we aim to investigate spatiotemporal clustering of individuals to uncover patterns of social interaction within a geographical context. By analysing tracking data from a group of individuals, we will identify and characterize social hotspots – locations where people are in close proximity at the same time. Our goal is to gain insights into the dynamics of these social hotspots, particularly how they are influenced by environmental factors such as weather, land use or other geographic features. Furthermore, we will incorporate a temporal dimension into our analysis, exploring how variations between time of day or day of week may affect the formation and intensity of these hotspots.

## Research Questions

-   How can we identify and characterize spatial patterns of social interactions from the tracking data of a group of individuals?

-   How do social hotspots change in response to environmental or temporal changes?

## Results / products

The final product of our work will be a map of social hotspots of the group of individuals studied. These might be identified locations, or they might be represented in the form of a heat map. Different maps with different subsets of the data may be produced and compared to analyse the effect of environmental and temporal factors on the social hotspots.

We expect the biggest social hotspot to be the ZHAW campus in Wädenswil, since the group of people being studied are all participants in the course that is being held there, which is also the course for which this project is being conducted. The Irchel campus, where some of the students of that course usually study, is also likely to be identified as a significant hotspot. Furthermore, certain train stations or routes on the way to this course could also be hotspots. It will be interesting to see if there will be recreational places identified at the end of workdays or on weekends, in cases where individuals spend time together outside of their studies.

## Data

<!-- What data will you use? Will you require additional context data? Where do you get this data from? Do you already have all the data? -->

We will use tracking data obtained from the pooled Posmo data of the students in this course. To examine the effect of various environmental factors, we will additionally consider data like land use and weather data in the analysis.

## Analytical concepts

<!-- Which analytical concepts will you use? What conceptual movement spaces and respective modelling approaches of trajectories will you be using? What additional spatial analysis methods will you be using? -->

The most important step of our analysis will consist in identifying meeting locations. We will do this using the methods introduced in the lecture, where we aim to temporally align the trajectories of individuals so that we can identify situations in which they were at the same place at the same time. Weighting these hotspots by the number of individuals involved in the encounter, we can then determine the relevance of the hotspots. Geographical and/or temporal classification of these hotspots will then allow us to analyse them in the context of our research questions.

## Python concepts

<!-- Which R concepts, functions, packages will you mainly use. What additional spatial analysis methods will you be using? -->

Possible packages to be used:

1.  Pandas: for data manipulation and analysis.

2.  Numpy: for working with arrays and matrices, filtering and cleaning movement data, and performing basic calculations on movement data.

3.  Matplotlib and seaborn: for data visualization and plotting.

4.  Folium: for interactive web map creation.

5.  Geopandas: for spatial data manipulation and analysis.

6.  Scikit-learn: for machine learning and spatial clustering.

7.   Scikit-mobility: a package specifically designed for movement data analysis.

8.  Trajectory: a package for preprocessing and analyzing spatio-temporal trajectories

## Risk analysis

<!-- What could be the biggest challenges/problems you might face? What is your plan B? -->

One conceptual difficulty will be determining how to define "meeting places" or "hotspots". Moreover, interpreting the context of how and why certain hotspots exist may prove to be challenging as we do not have any additional information on the travelers except their movement data, and the patterns may have multiple possible explanations. So it is important to consider the limitation of the data.

The quality of the data is also very important. Temporal lags or gaps in the data when the App does not record any data decrease the size of the dataset, which may make it more difficulty to detect such hotspots.

## Questions?

<!-- Which questions would you like to discuss at the coaching session? -->
