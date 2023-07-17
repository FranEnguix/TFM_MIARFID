from rest_framework import serializers
from .models import WeatherMeasure

class WeatherMeasureSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = WeatherMeasure
        fields = ('timestamp',)