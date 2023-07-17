import urllib.request, json

from django.shortcuts import render

from rest_framework import viewsets
from rest_framework.response import Response

from .serializers import WeatherMeasureSerializer
from .models import WeatherMeasure

from datetime import datetime, timedelta

def populate_data():
    ResultBytes = urllib.request.urlopen("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Valencia?unitGroup=metric&include=hours&key=6EGBGGGX8H6TDDNYE5WMW9CAH&contentType=json")
    jsonData = json.load(ResultBytes)
    days = [[datetime.fromtimestamp(d["datetimeEpoch"]), d["hours"]] for d in jsonData['days']]
    for day_dt, hours in days:
        for hour in hours:
            dt = datetime.fromtimestamp(hour["datetimeEpoch"])
            modified_dt = dt - timedelta(days=2)
            temperature = float(hour["temp"])
            humidity = int(hour["humidity"])
            pressure = int(hour["pressure"])
            weather, is_created = WeatherMeasure.objects.get_or_create(timestamp=modified_dt, defaults={'temperature': temperature, 'humidity': humidity, 'pressure': pressure})

# def dates_spatied_one_hour(date1, date2): 
#     diff = date1 - date2 # date1 > date2
#     days, seconds = diff.days, diff.seconds
#     hours = days * 24 + seconds // 3600
#     minutes = (seconds % 3600) // 60
#     seconds = seconds % 60
#     return hours < 2

class WeatherMeasureViewSet(viewsets.ModelViewSet):
    queryset = WeatherMeasure.objects.all().order_by('-timestamp')[:24]
    serializer_class = WeatherMeasureSerializer
    def list(self, response):
        lowest_pressure = 850
        highest_pressure = 1090
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        measures = WeatherMeasure.objects.filter(timestamp__range=[yesterday.strftime("%Y-%m-%dT%H"), now.strftime("%Y-%m-%dT%H")]).order_by('-timestamp')[:24]
        if len(measures) < 24:
            populate_data()

        measures = WeatherMeasure.objects.filter(timestamp__range=[yesterday.strftime("%Y-%m-%dT%H"), now.strftime("%Y-%m-%dT%H")]).order_by('-timestamp')[:24]

        api_data = []
        for measure in measures:
            api_data.insert(0, measure.pressure / (highest_pressure - lowest_pressure))     # 3
            api_data.insert(0, measure.humidity / 100)      # 2
            api_data.insert(0, measure.temperature / 60)    # 1
        

        return Response(json.dumps(api_data))
