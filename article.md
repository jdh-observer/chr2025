---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
## Birds ?
 
```{image} img/bird.jpeg
:width: 200px
:align: left
```

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Iuvaret mei, eam simul partem In nominati mel possit definiebas latine id? partem ipsum. Has corrumpit autem at convenire. deleniti Has rationibus ea at dolor deleniti latine ipsum. Copiosae sit simul amet, unum mel simul convenire. eam sit ea senserit possit Vis. Choro no, id? choro corrumpit ea, deleniti cu. definiebas In ea, id? sit at. Qui! has convenire. ea no, Vis in tincidunt! ea, tincidunt! ipsum dolor unum efficiendi,. Qui! ipsum iuvaret autem instructior unum vidisse deleniti vim amet, ea, nominati senserit Has.


## Birds

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Iuvaret mei, eam simul partem In nominati mel possit definiebas latine id? partem ipsum. Has corrumpit autem at convenire. deleniti Has rationibus ea at dolor deleniti latine ipsum. Copiosae sit simul amet, unum mel simul convenire. eam sit ea senserit possit Vis. Choro no, id? choro corrumpit ea, deleniti cu. definiebas In ea, id? sit at. Qui! has convenire. ea no, Vis in tincidunt! ea, tincidunt! ipsum dolor unum efficiendi,. Qui! ipsum iuvaret autem instructior unum vidisse deleniti vim amet, ea, nominati senserit Has.


## Packages and data

### Packages list

The following python packages are used in this notebook :

- `rdflib` : 
- `networkx` : 
- `matplotlib` : 

You can install the needeed packages with the following code :

```{code} shell
pip install session_info rdflib networkx matplotlib
```

Pour charger les librairies :
<!-- #endregion -->

```python
import session_info
```

```python
from rdflib import Graph, Literal, RDF, URIRef
# rdflib knows about quite a few popular namespaces, like W3C ontologies, schema.org etc.
from rdflib.namespace import FOAF , XSD

# Create a Graph
g = Graph()

# Create an RDF URI node to use as the subject for multiple triples
donna = URIRef("http://example.org/donna")

# Add triples using store's add() method.
g.add((donna, RDF.type, FOAF.Person))
g.add((donna, FOAF.nick, Literal("donna", lang="en")))
g.add((donna, FOAF.name, Literal("Donna Fales")))
g.add((donna, FOAF.mbox, URIRef("mailto:donna@example.org")))

# Add another person
ed = URIRef("http://example.org/edward")

# Add triples using store's add() method.
g.add((ed, RDF.type, FOAF.Person))
g.add((ed, FOAF.nick, Literal("ed", datatype=XSD.string)))
g.add((ed, FOAF.name, Literal("Edward Scissorhands")))
g.add((ed, FOAF.mbox, Literal("e.scissorhands@example.org", datatype=XSD.anyURI)))

# Bind the FOAF namespace to a prefix for more readable output
g.bind("foaf", FOAF)

# print all the data in the Notation3 format
print(g.serialize(format='n3'))
```

```python
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
import networkx as nx
import matplotlib.pyplot as plt

ug = rdflib_to_networkx_graph(g)
nx.draw(ug)
```

```python
import plotly.graph_objects as go
import numpy as np

# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for step in np.arange(0, 5, 0.1):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            name="𝜈 = " + str(step),
            x=np.arange(0, 10, 0.01),
            y=np.sin(step * np.arange(0, 10, 0.01))))

# Make 10th trace visible
fig.data[10].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()
```

# SPARQL Query


From here:
https://database.factgrid.de/wiki/FactGrid:Sample_queries

by taking the example: 
https://database.factgrid.de/query/#%23defaultView%3AMap%0ASELECT%20%3Fitem%20%3FitemLabel%20%3FOrt%20%3FOrtLabel%20%3FGeokoordinaten%20WHERE%20%7B%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%20bd%3AserviceParam%20wikibase%3Alanguage%20%22%5BAUTO_LANGUAGE%5D%2Cen%22.%20%7D%0A%20%20%3Fitem%20wdt%3AP2%20wd%3AQ10671.%0A%20%20%3Fitem%20wdt%3AP95%20%3FOrt.%0A%20%20%3FOrt%20wdt%3AP48%20%3FGeokoordinaten.%0A%20%20%3Fitem%20wdt%3AP97%20wd%3AQ10677.%0A%7D

```python
from IPython.display import Image, display
metadata={
    "jdh": {
        "module": "object",
        "object": {
            "type":"image",
            "source": ["Diachronic evolution of agency"]
        }
    }
}
display(Image("./media/factgrid_code.png"), metadata=metadata)
```

```python
sparql_query = """
SELECT ?item ?itemLabel ?Ort ?OrtLabel ?Geokoordinaten 
WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
  ?item wdt:P2 wd:Q10671.
  ?item wdt:P95 ?Ort.
  ?Ort wdt:P48 ?Geokoordinaten.
  ?item wdt:P97 wd:Q10677.
}
"""
```

```python
import sys
from SPARQLWrapper import SPARQLWrapper, JSON

endpoint_url = "https://database.factgrid.de/sparql"

query = """#defaultView:Map
SELECT ?item ?itemLabel ?Ort ?OrtLabel ?Geokoordinaten WHERE {
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
  ?item wdt:P2 wd:Q10671.
  ?item wdt:P95 ?Ort.
  ?Ort wdt:P48 ?Geokoordinaten.
  ?item wdt:P97 wd:Q10677.
}"""


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


results = get_results(endpoint_url, query)
```

```python
for result in results["results"]["bindings"][:5]:
    print(result)

```

## Map

```python
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import re

# Parse the SPARQL bindings
bindings = results["results"]["bindings"]

data = []
for result in bindings:
    wkt = result['Geokoordinaten']['value']
    
    # Extract lon, lat from WKT Point format
    match = re.search(r'Point\(([\d.]+)\s+([\d.]+)\)', wkt)
    if match:
        lon, lat = float(match.group(1)), float(match.group(2))

        data.append({
            'item': result['item']['value'],
            'itemLabel': result['itemLabel']['value'],
            'location': result['OrtLabel']['value'],
            'latitude': lat,
            'longitude': lon
        })

df = pd.DataFrame(data)

# Create the map centered on the mean coordinates
map_center = [df['latitude'].mean(), df['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=6)

# Add marker cluster
marker_cluster = MarkerCluster().add_to(m)

# Add markers
for _, row in df.iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=f"<b>{row['location']}</b><br>{row['itemLabel']}",
        tooltip=row['location']
    ).add_to(marker_cluster)

# Save and display
m.save('letters_map.html')


```

```python
from IPython.display import IFrame
IFrame("letters_map.html", width="100%", height="600px")
```

```python
session_info.show()
```

## Session information
