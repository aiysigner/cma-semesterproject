# Proposal for Semester Project

**Patterns & Trends in Environmental Data / Computational Movement Analysis Geo 880**

| Semester:      | FS23                            |
|:---------------|:--------------------------------|
| **Data:**      | Posmo Location of Meeting Spots |
| **Title:**     | Hot Spot Analysis               |
| **Student 1:** | Diego Gomes                     |
| **Student 2:** | Aiyana Signer                   |

## Abstract

As a Hotspot Analysis, the objective is to identify and characterize potential meeting spots students from this converge based on the pooled Posmo data. This analysis will utilize Meets calculation to detect meeting hotspots and examine temporal variations in the frequency of these gatherings. This project aims to contribute to a more comprehensive understanding of student social networks and their spatial dynamics.

## Research Questions

-   Can common meeting hotspots be efficiently and effectively detected based on pooled posmo data of students of the CMA course?

-   Are there any notable spatial patterns in the distribution of meeting hotspots across different regions or neighborhoods?

-   How do the hotspots vary over time (e.g., day vs. evening, weekday vs. weekend)?

## Results / products

It is anticipated that the Wädenswil and Irchel university campuses will serve as hotspots for synchronous student interactions, due to participation in the same courses. It is also possible that train stations will pass as meeting points as they are central hubs of transportation.

A heat map could be produced, depicting the various hotspots, with the most frequented areas appearing more salient. Temporal patterns could be analyzed through the use of a time-space cube. One could also focusing on a specific hotspot, and generate a histogram or line graph to demonstrate peak utilization periods (i.e., time of day or day of the week).

## Data

<!-- What data will you use? Will you require additional context data? Where do you get this data from? Do you already have all the data? -->

We will use movement data provided by the pooled Posmo data from the students in this course. To enhance the analysis, we could also use land use and land cover data to analyze if most hot spots are indeed more likely in urban areas compared to more natural areas.

## Analytical concepts

<!-- Which analytical concepts will you use? What conceptual movement spaces and respective modelling approaches of trajectories will you be using? What additional spatial analysis methods will you be using? -->

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

Conceptually, a difficulty is how "meetings" or "hotspots" should be defined. In addition, interpreting the context of how and why certain hotspots exist may prove to be challenging as we do not have any additional information on the travelers except their movement data, and the patterns may have multiple possible explanations. So it is important to consider the limitation of the data.

The quality of the data is also very important. Temporal lags or gaps in the data when the App does not record any data decrease the size of the dataset, which may make it more difficulty to detect such hotspots.

## Questions?

<!-- Which questions would you like to discuss at the coaching session? -->
