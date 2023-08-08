# From: gluonts/src/gluonts/time_feature/_base.py
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    # __call)) 클래스의 인스턴스를 함수처럼 호출하는 특수 메서드
    # pd.DatatimeIndex를 Input으로 받아서 Numpy 배열 반복
    # 이는 기본 Class에서 구현되지 않고, 하위 클래스에서 재정의 해야 함.
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    # __repr__는 객체의 문자열 표현을 반환하는 (사람이 보기에 가장 편한 형태로) 특수 메서드
    # Class 이름을 문자열로 반환하고 "()"를 더하여 Callarble한 객체
    def __repr__(self):
        return self.__class__.__name__ + "()"

# pd.DatatimeIndex를 기반으로 

# SecondOfMinute는 분의 초를 -0.5와 0.5 사이의 값으로 나타내는 TimeFeature의 하위 클래스
# __call__ method는 주어진 DatetimeIndex 객체에 대한 초/분(1분의 최대 초)을 계산.
class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5
    
"""
index가 타임스탬프 [10:15:20, 10:15:30, 10:15:40]을 나타내는 경우 index.second는 [20, 30, 40]을 반환
index.second / 59.0 - 0.5' 계산은 초 값을 -0.5에서 0.5 사이의 범위로 조정.
이후 59.0으로 나누면 0과 1 사이의 값이 정규화되고 0.5를 빼면 범위가 -0.5와 0.5로 이동
"""

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    
    빈도를 입력으로 받아서, 해당 빈도에 적합한 Time Feature class 목록 반환
    
    """

    # features_by_offsets은 다양한 오프셋(예: offsets.YearEnd, offsets.Minute)을 해당 시간 피처 클래스에 매핑
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    # freq_str을 offset으로 변환
    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
# time_features 함수는 pd.DatetimeIndex, 빈도 문자열(freq) ('h')를 Input으로 받음
# 계산된 특성 값은 'np.vstack'을 사용하여 수직으로 쌓여서 각 행이 다른 시간 특성을 나타내고 
# 각 열이 '날짜'의 특정 타임스탬프에 대한 특성 값을 나타냄