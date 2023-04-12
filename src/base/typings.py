#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:58:01 2022

@author: student
"""
import json
from dataclasses import dataclass


@dataclass
class SampleArea:
    xmin: int
    xmax: int
    ymin: int
    ymax: int
        
    def width(self):
        return self.xmax - self.xmin
    
    def height(self):
        return self.ymax - self.ymin
    
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)