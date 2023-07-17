from django.db import models

class WeatherMeasure(models.Model):
    timestamp = models.DateTimeField(primary_key=True)
    temperature = models.FloatField(null=True) # Celsius
    humidity = models.IntegerField(null=True)
    pressure = models.IntegerField(null=True) # hPa 
    def __str__(self):
        return f"{self.temperature}_{self.humidity}_{self.pressure}"
