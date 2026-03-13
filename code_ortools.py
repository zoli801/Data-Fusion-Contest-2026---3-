#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
    ОПТИМИЗАЦИЯ МАРШРУТОВ ГЕРОЕВ - РЕШЕНИЕ ЗАДАЧИ VRPTW
================================================================================

    Автор: Data Fusion Contest 2026
    Описание: Алгоритм для оптимизации сбора золота героями
              с учетом временных окон и ограничений на перемещение

    Версия: 2.0 (с улучшенной читаемостью)
================================================================================
"""

import os
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any, Callable
from datetime import datetime
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


# =================================================================================
# БЛОК 1: КОНСТАНТЫ И ПАРАМЕТРЫ ИГРЫ
# =================================================================================

class GameConstants:
    """
    Класс-контейнер для всех игровых констант.

    Эти параметры заданы условиями задачи и не должны изменяться,
    так как от них зависит корректность симуляции и итоговый счет.
    """

    # ========== БАЗОВЫЕ ПАРАМЕТРЫ ==========
    VISIT_COST: int = 100  # Стоимость посещения мельницы (очки хода)
    HERO_COST: int = 2500  # Стоимость найма одного героя (золото)
    NUM_DAYS: int = 7  # Количество дней в игровой неделе
    NUM_OBJECTS: int = 700  # Общее количество мельниц на карте

    # ========== ПАРАМЕТРЫ ОПТИМИЗАЦИИ ==========
    BASE_HEROES: int = 20  # Базовое количество героев для первого дня
    SKIP_PENALTY: int = 50000  # Штраф за пропуск мельницы (для OR-Tools)
    FIXED_COST: int = 100000  # Фиксированная стоимость использования героя
    LATE_PENALTY: int = 100  # Штраф за опоздание (добавляется к расстоянию)

    # ========== ЛИМИТЫ ВРЕМЕНИ ==========
    TIME_DAY1: int = 120  # Секунд на оптимизацию первого дня
    TIME_OTHER_DAYS: int = 10  # Секунд на оптимизацию остальных дней


class FilePaths:
    """
    Класс-контейнер для путей к файлам данных.

    Все файлы должны находиться в той же директории, что и скрипт.
    """

    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

    HEROES: str = os.path.join(BASE_DIR, "data_heroes.csv")
    OBJECTS: str = os.path.join(BASE_DIR, "data_objects.csv")
    DIST_START: str = os.path.join(BASE_DIR, "dist_start.csv")
    DIST_OBJECTS: str = os.path.join(BASE_DIR, "dist_objects.csv")
    SUBMISSION: str = os.path.join(BASE_DIR, "submission.csv")


# =================================================================================
# БЛОК 2: СТРУКТУРЫ ДАННЫХ
# =================================================================================

class GameData:
    """
    Класс для хранения всех загруженных данных игры.

    Attributes:
        heroes (pd.DataFrame): Данные о героях (ID и очки хода)
        objects (pd.DataFrame): Данные о мельницах (ID, день, награда)
        start_distances (Dict[int, int]): Расстояния от замка до мельниц
        distance_matrix (np.ndarray): Матрица расстояний между мельницами
    """

    def __init__(self, heroes: pd.DataFrame, objects: pd.DataFrame,
                 start_distances: Dict[int, int], distance_matrix: np.ndarray):
        self.heroes = heroes
        self.objects = objects
        self.start_distances = start_distances
        self.distance_matrix = distance_matrix

        # Проверка целостности данных
        self._validate_data()

    def _validate_data(self) -> None:
        """Проверяет, что все данные загружены корректно."""
        assert len(self.heroes) > 0, "Нет данных о героях"
        assert len(self.objects) == 700, f"Ожидалось 700 объектов, получено {len(self.objects)}"
        assert self.distance_matrix.shape == (
        700, 700), f"Матрица расстояний должна быть 700x700, получено {self.distance_matrix.shape}"

    def get_hero_move_points(self, hero_id: int) -> int:
        """Возвращает дневной запас очков хода героя."""
        return int(self.heroes[self.heroes["hero_id"] == hero_id]["move_points"].iloc[0])

    def get_object_info(self, object_id: int) -> Tuple[int, int]:
        """Возвращает (день_открытия, награда) для мельницы."""
        obj = self.objects[self.objects["object_id"] == object_id].iloc[0]
        return int(obj["day_open"]), int(obj["reward"])

    def get_objects_by_day(self, day: int) -> List[int]:
        """Возвращает список ID мельниц, доступных в указанный день."""
        return sorted(self.objects[self.objects["day_open"] == day]["object_id"].tolist())


class SimulationResult:
    """
    Класс для хранения результатов симуляции маршрута.

    Attributes:
        score (int): Итоговый счет (собранное золото минус затраты)
        reward (int): Собранное золото
        hero_cost (int): Затраты на наем героев
        max_hero (int): Максимальный ID использованного героя
        unique_objects (int): Количество уникальных посещенных мельниц
    """

    def __init__(self, score: int, reward: int, hero_cost: int, max_hero: int):
        self.score = score
        self.reward = reward
        self.hero_cost = hero_cost
        self.max_hero = max_hero
        self.unique_objects = 0  # Будет установлено позже

    @classmethod
    def from_dict(cls, data: Dict) -> 'SimulationResult':
        """Создает объект из словаря."""
        return cls(
            score=data.get('score', 0),
            reward=data.get('reward', 0),
            hero_cost=data.get('hero_cost', 0),
            max_hero=data.get('max_hero', 0)
        )

    def __str__(self) -> str:
        return (f"Счет: {self.score:,} | Золото: {self.reward:,} | "
                f"Герои: {self.max_hero} | Объекты: {self.unique_objects}")


# =================================================================================
# БЛОК 3: ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =================================================================================

def load_game_data() -> GameData:
    """
    Загружает все необходимые данные из CSV файлов.

    Процесс загрузки:
    1. Чтение CSV файлов с помощью pandas
    2. Извлечение матрицы расстояний из dist_objects.csv
    3. Создание словаря расстояний от старта
    4. Валидация загруженных данных

    Returns:
        GameData: Структурированные данные для оптимизации

    Raises:
        FileNotFoundError: Если какой-либо файл не найден
        ValueError: Если данные имеют неверный формат
    """
    print("\n📂 Загрузка данных...")

    try:
        # Загрузка CSV файлов
        heroes = pd.read_csv(FilePaths.HEROES)
        objects = pd.read_csv(FilePaths.OBJECTS)
        dist_start = pd.read_csv(FilePaths.DIST_START)
        dist_objects = pd.read_csv(FilePaths.DIST_OBJECTS)

        print(f"   ✓ Герои: {len(heroes)} шт.")
        print(f"   ✓ Мельницы: {len(objects)} шт.")
        print(f"   ✓ Расстояния от старта: {len(dist_start)} шт.")
        print(f"   ✓ Матрица расстояний: {dist_objects.shape}")

        # Извлечение матрицы расстояний между объектами
        object_columns = [col for col in dist_objects.columns if col.startswith("object_")]
        distance_matrix = dist_objects[object_columns].values.astype(np.int64)

        # Создание словаря расстояний от старта
        start_distances = dict(zip(dist_start["object_id"], dist_start["dist_start"]))

        return GameData(heroes, objects, start_distances, distance_matrix)

    except FileNotFoundError as e:
        print(f"❌ Ошибка: файл не найден - {e}")
        raise
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        raise


def get_distance_between_points(data: GameData, point_a: int, point_b: int) -> int:
    """
    Вычисляет расстояние между двумя точками на карте.

    Правила расчета:
    - Если точка A = 0 (замок), используется dist_start до точки B
    - Если точка B = 0 (замок), используется dist_start от точки A
    - Иначе используется матрица расстояний между объектами

    Args:
        data: Данные игры
        point_a: ID начальной точки (0 для замка)
        point_b: ID конечной точки (0 для замка)

    Returns:
        int: Расстояние в очках хода
    """
    if point_a == 0:
        return data.start_distances.get(point_b, 0)
    if point_b == 0:
        return data.start_distances.get(point_a, 0)
    return data.distance_matrix[point_a - 1, point_b - 1]


# =================================================================================
# БЛОК 4: СИМУЛЯЦИЯ МАРШРУТОВ
# =================================================================================

class RouteSimulator:
    """
    Класс для симуляции прохождения маршрутов героями.

    Учитывает все игровые механики:
    - Дневной лимит очков хода
    - Ожидание дня открытия мельницы
    - Правило "последнего шага"
    - Глобальный счетчик дней
    - Штрафы за опоздание
    """

    def __init__(self, data: GameData):
        self.data = data

    def simulate(self, solution: List[Tuple[int, int]]) -> SimulationResult:
        """
        Полная симуляция маршрута всех героев.

        Алгоритм симуляции:
        1. Группировка объектов по героям
        2. Для каждого героя последовательное прохождение маршрута
        3. Учет перемещений между днями
        4. Расчет наград и штрафов

        Args:
            solution: Список пар (hero_id, object_id) в порядке посещения

        Returns:
            SimulationResult: Результаты симуляции
        """
        if not solution:
            return SimulationResult(
                score=-GameConstants.HERO_COST,
                reward=0,
                hero_cost=0,
                max_hero=0
            )

        # Группировка объектов по героям
        routes = self._group_by_hero(solution)
        max_hero = max(routes.keys())

        # Симуляция для каждого героя
        total_reward = 0
        for hero_id in sorted(routes.keys()):
            hero_reward = self._simulate_hero_route(hero_id, routes[hero_id])
            total_reward += hero_reward

        # Расчет итогового счета
        hero_cost = max_hero * GameConstants.HERO_COST
        score = total_reward - hero_cost

        result = SimulationResult(score, total_reward, hero_cost, max_hero)
        result.unique_objects = len(set(obj for _, obj in solution))

        return result

    def _group_by_hero(self, solution: List[Tuple[int, int]]) -> Dict[int, List[int]]:
        """Группирует объекты по ID героя."""
        routes = defaultdict(list)
        for hero_id, object_id in solution:
            routes[hero_id].append(object_id)
        return dict(routes)

    def _simulate_hero_route(self, hero_id: int, objects: List[int]) -> int:
        """
        Симулирует маршрут одного героя.

        Возвращает сумму наград за успешные посещения.
        """
        move_points = self.data.get_hero_move_points(hero_id)

        current_pos = 0  # Начинаем из замка
        remaining = move_points
        current_day = 1
        hero_reward = 0

        for obj_id in objects:
            day_open, reward = self.data.get_object_info(obj_id)

            # Расстояние до следующего объекта
            distance = get_distance_between_points(self.data, current_pos, obj_id)

            # Перемещение с учетом смены дней
            current_day, remaining = self._travel_to_object(
                distance, remaining, move_points, current_day
            )

            # Если неделя закончилась, прекращаем движение
            if current_day > GameConstants.NUM_DAYS:
                break

            # Ожидание дня открытия
            if current_day < day_open:
                current_day = day_open
                remaining = move_points

            # Посещение мельницы (с учетом last move rule)
            remaining = self._visit_object(remaining)

            # Начисление награды, если посещение вовремя
            if current_day == day_open:
                hero_reward += reward

            current_pos = obj_id

        return hero_reward

    def _travel_to_object(self, distance: int, remaining: int,
                          move_points: int, current_day: int) -> Tuple[int, int]:
        """
        Симулирует перемещение к объекту с учетом смены дней.

        Returns:
            Tuple[int, int]: (новый_день, оставшиеся_очки)
        """
        while current_day <= GameConstants.NUM_DAYS:
            if distance <= remaining:
                break
            distance -= remaining
            current_day += 1
            remaining = move_points

        return current_day, remaining

    def _visit_object(self, remaining: int) -> int:
        """
        Симулирует посещение мельницы.

        Учитывает правило "последнего шага":
        если осталось меньше VISIT_COST очков,
        посещение все равно успешно.
        """
        if remaining >= GameConstants.VISIT_COST:
            return remaining - GameConstants.VISIT_COST
        return 0


# =================================================================================
# БЛОК 5: ПОСТРОЕНИЕ МАТРИЦ РАССТОЯНИЙ
# =================================================================================

class DistanceMatrixBuilder:
    """
    Класс для построения и модификации матриц расстояний.

    Отвечает за:
    - Создание полной матрицы с замком (индекс 0)
    - Добавление стоимости посещения
    - Создание подматриц для подмножеств объектов
    """

    def __init__(self, data: GameData):
        self.data = data
        self.full_matrix = self._build_full_matrix()

    def _build_full_matrix(self) -> np.ndarray:
        """
        Строит полную матрицу расстояний размером (701 x 701).

        Индекс 0 соответствует замку.
        Индексы 1-700 соответствуют объектам 1-700.
        """
        num_objects = GameConstants.NUM_OBJECTS
        matrix = np.zeros((num_objects + 1, num_objects + 1), dtype=np.int64)

        # Расстояния от/до замка
        for j in range(1, num_objects + 1):
            dist = self.data.start_distances.get(j, 0)
            matrix[0, j] = matrix[j, 0] = dist

        # Расстояния между объектами
        matrix[1:, 1:] = self.data.distance_matrix

        return matrix

    def add_visit_cost(self, matrix: np.ndarray, depot_idx: int = 0) -> None:
        """
        Добавляет стоимость посещения ко всем перемещениям.

        Args:
            matrix: Матрица для модификации (изменяется на месте)
            depot_idx: Индекс замка (обычно 0)
        """
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if j == depot_idx:
                    matrix[i][j] = 0  # Возврат в замок бесплатный
                else:
                    matrix[i][j] = int(matrix[i][j]) + GameConstants.VISIT_COST

    def create_submatrix(self, object_ids: List[int]) -> np.ndarray:
        """
        Создает подматрицу для указанных объектов.

        Args:
            object_ids: Список ID объектов для включения

        Returns:
            np.ndarray: Подматрица с замком (индекс 0) и объектами
        """
        indices = [0] + object_ids
        return self.full_matrix[np.ix_(indices, indices)].copy()

    def create_vehicle_matrix(self, start_pos: int, day_objects: List[int]) -> np.ndarray:
        """
        Создает матрицу для конкретного героя с учетом его позиции.

        Args:
            start_pos: Текущая позиция героя (0 для замка)
            day_objects: Объекты, доступные в текущий день

        Returns:
            np.ndarray: Матрица для оптимизации маршрута героя
        """
        # Вычисляем расстояния от текущей позиции
        if start_pos == 0:
            distances = np.array([self.data.start_distances.get(i, 0)
                                  for i in range(1, GameConstants.NUM_OBJECTS + 1)],
                                 dtype=np.int64)
        else:
            distances = self.data.distance_matrix[start_pos - 1, :].copy()

        # Строим полную матрицу
        num_objects = GameConstants.NUM_OBJECTS
        full = np.zeros((num_objects + 1, num_objects + 1), dtype=np.int64)

        for j in range(1, num_objects + 1):
            full[0, j] = full[j, 0] = distances[j - 1]

        full[1:, 1:] = self.data.distance_matrix

        # Создаем подматрицу для объектов дня
        indices = [0] + day_objects
        submatrix = full[np.ix_(indices, indices)].copy()
        self.add_visit_cost(submatrix)

        return submatrix


# =================================================================================
# БЛОК 6: ОПТИМИЗАЦИЯ ПЕРВОГО ДНЯ
# =================================================================================

class Day1Optimizer:
    """
    Оптимизатор маршрутов для первого дня.

    Особенности первого дня:
    - Все герои стартуют из замка
    - Можно выбрать любое количество героев
    - Объекты только первого дня
    """

    def __init__(self, data: GameData, matrix_builder: DistanceMatrixBuilder):
        self.data = data
        self.matrix_builder = matrix_builder

    def optimize(self, num_heroes: int = None, time_limit: int = None,
                 penalty: int = None) -> Tuple[List[List[int]], Dict[int, int]]:
        """
        Запускает оптимизацию для первого дня.

        Returns:
            Tuple[List[List[int]], Dict[int, int]]:
                - Список маршрутов (индексы в подматрице)
                - Словарь последних позиций героев
        """
        # Получаем объекты первого дня
        day1_objects = self.data.get_objects_by_day(1)
        if not day1_objects:
            return [], {}

        # Подготовка данных
        submatrix = self.matrix_builder.create_submatrix(day1_objects)
        self.matrix_builder.add_visit_cost(submatrix)
        submatrix_list = submatrix.tolist()

        # Настройка параметров
        num_heroes = num_heroes or GameConstants.BASE_HEROES
        heroes_subset = self.data.heroes.sort_values("hero_id").head(num_heroes)
        capacities = (heroes_subset["move_points"] + GameConstants.LATE_PENALTY).tolist()

        # Создание модели OR-Tools
        manager = pywrapcp.RoutingIndexManager(len(submatrix_list), num_heroes, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Callback для расстояний
        def distance_cb(from_idx, to_idx):
            from_node = manager.IndexToNode(from_idx)
            to_node = manager.IndexToNode(to_idx)
            return submatrix_list[from_node][to_node]

        transit_cb = routing.RegisterTransitCallback(distance_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

        # Штрафы за пропущенные объекты
        penalty_value = penalty or GameConstants.SKIP_PENALTY
        for node in range(1, len(submatrix_list)):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty_value)

        # Ограничение на дневной пробег
        routing.AddDimensionWithVehicleCapacity(
            transit_cb, 0, capacities, True, "Distance"
        )

        # Параметры поиска
        search_params = self._create_search_params(time_limit or GameConstants.TIME_DAY1)

        # Решение
        solution = routing.SolveWithParameters(search_params)
        if not solution:
            return [], {}

        # Извлечение маршрутов
        routes = self._extract_routes(routing, manager, solution, num_heroes)

        # Фильтрация пустых маршрутов
        routes = [r for r in routes if len(r) > 1 and set(r) - {0}]

        # Определение последних позиций
        last_positions = self._get_last_positions(routes, day1_objects)

        return routes, last_positions

    def _create_search_params(self, time_limit: int):
        """Создает параметры для поиска решения."""
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        params.time_limit.seconds = time_limit
        return params

    def _extract_routes(self, routing, manager, solution, num_heroes):
        """Извлекает маршруты из решения OR-Tools."""
        routes = []
        for vehicle_id in range(num_heroes):
            idx = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(idx):
                route.append(manager.IndexToNode(idx))
                idx = solution.Value(routing.NextVar(idx))
            routes.append(route)
        return routes

    def _get_last_positions(self, routes, day1_objects):
        """Определяет последние позиции героев после первого дня."""
        last_positions = {}
        for hero_id, route in enumerate(routes):
            last_node = 0
            for node in reversed(route):
                if node != 0:
                    last_node = node
                    break

            if last_node > 0 and last_node <= len(day1_objects):
                last_positions[hero_id] = day1_objects[last_node - 1]
            else:
                last_positions[hero_id] = 0

        return last_positions


# =================================================================================
# БЛОК 7: ОПТИМИЗАЦИЯ ПОСЛЕДУЮЩИХ ДНЕЙ
# =================================================================================

class DayNOptimizer:
    """
    Оптимизатор маршрутов для дней 2-7.

    Особенности:
    - У каждого героя своя стартовая позиция
    - Разные матрицы расстояний для разных героев
    - Учет завершивших маршрут героев
    """

    def __init__(self, data: GameData, matrix_builder: DistanceMatrixBuilder):
        self.data = data
        self.matrix_builder = matrix_builder

    def optimize(self, day: int, last_positions: Dict[int, int],
                 prev_routes: List[List[int]], time_limit: int = None) -> Tuple[List[List[int]], Dict[int, int]]:
        """
        Запускает оптимизацию для указанного дня.

        Args:
            day: Номер дня (2-7)
            last_positions: Позиции героев на начало дня
            prev_routes: Маршруты предыдущего дня
            time_limit: Лимит времени на оптимизацию

        Returns:
            Tuple[List[List[int]], Dict[int, int]]: Маршруты и новые позиции
        """
        num_heroes = len(last_positions)
        day_objects = self.data.get_objects_by_day(day)

        if not day_objects:
            return [[] for _ in range(num_heroes)], dict(last_positions)

        # Создание матриц для каждого героя
        vehicle_matrices = self._create_vehicle_matrices(
            last_positions, prev_routes, day_objects
        )

        if not vehicle_matrices:
            return [[] for _ in range(num_heroes)], dict(last_positions)

        # Оптимизация
        routes, new_positions = self._solve_multi_matrix(
            vehicle_matrices, num_heroes, day_objects, last_positions, time_limit
        )

        return routes, new_positions

    def _create_vehicle_matrices(self, last_positions: Dict[int, int],
                                 prev_routes: List[List[int]],
                                 day_objects: List[int]) -> List[List[List[int]]]:
        """Создает отдельные матрицы для каждого героя."""
        # Определяем завершивших маршрут
        finished = {}
        if prev_routes:
            for hero_id, route in enumerate(prev_routes):
                finished[hero_id] = len(route) <= 1

        matrices = []
        for hero_id in range(len(last_positions)):
            start = last_positions.get(hero_id, 0)

            if finished.get(hero_id, False):
                # Герой уже в замке - нулевые расстояния
                distances = np.zeros(GameConstants.NUM_OBJECTS, dtype=np.int64)
            elif start == 0:
                # Герой в замке - расстояния от старта
                distances = np.array([self.data.start_distances.get(i, 0)
                                      for i in range(1, GameConstants.NUM_OBJECTS + 1)],
                                     dtype=np.int64)
            else:
                # Герой у объекта - расстояния оттуда
                distances = self.data.distance_matrix[start - 1, :].copy()

            # Строим матрицу
            matrix = self._build_matrix_from_distances(distances, day_objects)
            matrices.append(matrix.tolist())

        return matrices

    def _build_matrix_from_distances(self, distances: np.ndarray,
                                     day_objects: List[int]) -> np.ndarray:
        """Строит матрицу на основе расстояний от текущей позиции."""
        num_objects = GameConstants.NUM_OBJECTS
        full = np.zeros((num_objects + 1, num_objects + 1), dtype=np.int64)

        for j in range(1, num_objects + 1):
            full[0, j] = full[j, 0] = distances[j - 1]

        full[1:, 1:] = self.data.distance_matrix

        # Подматрица для объектов дня
        indices = [0] + day_objects
        submatrix = full[np.ix_(indices, indices)].copy()
        self.matrix_builder.add_visit_cost(submatrix)

        return submatrix

    def _solve_multi_matrix(self, vehicle_matrices: List, num_heroes: int,
                            day_objects: List[int], last_positions: Dict[int, int],
                            time_limit: int) -> Tuple[List[List[int]], Dict[int, int]]:
        """Решает задачу с разными матрицами для разных героев."""
        matrix_size = len(vehicle_matrices[0])

        # Создание модели
        manager = pywrapcp.RoutingIndexManager(matrix_size, num_heroes, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Регистрация callback'ов для каждого героя
        callbacks = []
        for v_id in range(num_heroes):
            def make_callback(vid):
                def callback(from_idx, to_idx):
                    from_node = manager.IndexToNode(from_idx)
                    to_node = manager.IndexToNode(to_idx)
                    return vehicle_matrices[vid][from_node][to_node]

                return callback

            cb = make_callback(v_id)
            cb_idx = routing.RegisterTransitCallback(cb)
            routing.SetArcCostEvaluatorOfVehicle(cb_idx, v_id)
            callbacks.append(cb_idx)

        # Фиксированная стоимость использования героя
        for v_id in range(num_heroes):
            routing.SetFixedCostOfVehicle(GameConstants.FIXED_COST, v_id)

        # Ограничения на дневной пробег
        heroes_subset = self.data.heroes.sort_values("hero_id").head(num_heroes)
        capacities = (heroes_subset["move_points"] + GameConstants.LATE_PENALTY).tolist()

        routing.AddDimensionWithVehicleTransitAndCapacity(
            callbacks, 0, capacities, True, "Distance"
        )

        # Параметры поиска
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        params.time_limit.seconds = time_limit or GameConstants.TIME_OTHER_DAYS

        # Решение
        solution = routing.SolveWithParameters(params)
        if not solution:
            return [[] for _ in range(num_heroes)], dict(last_positions)

        # Извлечение маршрутов
        routes = []
        for v_id in range(num_heroes):
            idx = routing.Start(v_id)
            route = []
            while not routing.IsEnd(idx):
                route.append(manager.IndexToNode(idx))
                idx = solution.Value(routing.NextVar(idx))
            routes.append(route)

        # Определение новых позиций
        new_positions = self._get_new_positions(
            routes, day_objects, last_positions
        )

        return routes, new_positions

    def _get_new_positions(self, routes: List[List[int]], day_objects: List[int],
                           last_positions: Dict[int, int]) -> Dict[int, int]:
        """Определяет новые позиции героев после дня."""
        new_positions = {}

        for v_id, route in enumerate(routes):
            if len(route) > 1:
                last_node = 0
                for node in reversed(route):
                    if node != 0:
                        last_node = node
                        break

                if last_node > 0 and last_node <= len(day_objects):
                    new_positions[v_id] = day_objects[last_node - 1]
                else:
                    new_positions[v_id] = last_positions.get(v_id, 0)
            else:
                new_positions[v_id] = last_positions.get(v_id, 0)

        return new_positions


# =================================================================================
# БЛОК 8: ЛОКАЛЬНЫЙ ПОИСК
# =================================================================================

class LocalSearchOptimizer:
    """
    Улучшение решения с помощью локального поиска.

    Стратегия: пытается переместить объекты от героя с большим ID
    к предыдущему герою, если это улучшает результат.
    """

    def __init__(self, simulator: RouteSimulator):
        self.simulator = simulator

    def improve(self, solution: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Улучшает решение с помощью итеративного локального поиска.

        Args:
            solution: Исходное решение

        Returns:
            List[Tuple[int, int]]: Улучшенное решение
        """
        current = solution
        iteration = 0

        while True:
            iteration += 1
            improved = self._try_improve(current)

            if improved == current:
                break

            current = improved

        return current

    def _try_improve(self, solution: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Пробует одно улучшение."""
        # Группировка по героям
        routes = defaultdict(list)
        for hero_id, obj_id in solution:
            routes[hero_id].append(obj_id)

        max_hero = max(routes.keys()) if routes else 0

        if max_hero <= 1:
            return solution

        # Перемещение объектов последнего героя
        moved_objects = routes.pop(max_hero, [])
        routes[max_hero - 1] = routes.get(max_hero - 1, []) + moved_objects

        # Сборка нового решения
        new_solution = []
        for hero_id in sorted(routes):
            for obj_id in routes[hero_id]:
                new_solution.append((hero_id, obj_id))

        # Сравнение результатов
        old_score = self.simulator.simulate(solution).score
        new_score = self.simulator.simulate(new_solution).score

        return new_solution if new_score > old_score else solution


# =================================================================================
# БЛОК 9: ОСНОВНОЙ ОПТИМИЗАТОР
# =================================================================================

class HeroRouteOptimizer:
    """
    Главный класс оптимизатора, координирующий весь процесс.

    Этапы работы:
    1. Загрузка данных
    2. Оптимизация по дням (день 1, затем дни 2-7)
    3. Локальное улучшение
    4. Сохранение результата
    """

    def __init__(self):
        self.data = load_game_data()
        self.matrix_builder = DistanceMatrixBuilder(self.data)
        self.simulator = RouteSimulator(self.data)
        self.local_search = LocalSearchOptimizer(self.simulator)

        # Компоненты оптимизации
        self.day1_optimizer = Day1Optimizer(self.data, self.matrix_builder)
        self.dayN_optimizer = DayNOptimizer(self.data, self.matrix_builder)

    def optimize(self, num_heroes: int = None, time_day1: int = None,
                 time_other: int = None, penalty: int = None,
                 verbose: bool = True) -> List[Tuple[int, int]]:
        """
        Запускает полную оптимизацию на все дни.

        Args:
            num_heroes: Количество героев для первого дня
            time_day1: Лимит времени на первый день
            time_other: Лимит времени на остальные дни
            penalty: Штраф за пропуск объекта
            verbose: Детальный вывод

        Returns:
            List[Tuple[int, int]]: Итоговое решение
        """
        if verbose:
            self._print_header(num_heroes)

        # День 1
        routes, positions = self.day1_optimizer.optimize(
            num_heroes=num_heroes,
            time_limit=time_day1,
            penalty=penalty
        )

        if not routes:
            if verbose:
                print("❌ Не удалось найти решение для первого дня")
            return []

        all_routes = [routes]

        if verbose:
            active = len(routes)
            print(f"✓ День 1: {active} героев в пути")

        # Дни 2-7
        for day in range(2, GameConstants.NUM_DAYS + 1):
            prev_routes = all_routes[-1]
            routes, positions = self.dayN_optimizer.optimize(
                day=day,
                last_positions=positions,
                prev_routes=prev_routes,
                time_limit=time_other
            )

            all_routes.append(routes)

            if verbose:
                active = sum(1 for r in routes if len(r) > 1)
                print(f"✓ День {day}: {active} героев в пути")

        # Преобразование в итоговое решение
        solution = self._routes_to_solution(all_routes)

        return solution

    def _routes_to_solution(self, all_routes: List[List[List[int]]]) -> List[Tuple[int, int]]:
        """Преобразует маршруты всех дней в финальное решение."""
        solution = []

        for day_idx, day_routes in enumerate(all_routes, start=1):
            day_objects = self.data.get_objects_by_day(day_idx)

            for hero_idx, route in enumerate(day_routes):
                objects = self._route_indices_to_objects(route, day_objects)
                hero_id = hero_idx + 1

                for obj_id in objects:
                    solution.append((hero_id, obj_id))

        return solution

    def _route_indices_to_objects(self, route: List[int], day_objects: List[int]) -> List[int]:
        """Преобразует индексы маршрута в ID объектов."""
        if not route or len(route) <= 1:
            return []

        return [day_objects[idx - 1] for idx in route
                if idx != 0 and 1 <= idx <= len(day_objects)]

    def _print_header(self, num_heroes: int = None):
        """Выводит заголовок для дня."""
        heroes = num_heroes or GameConstants.BASE_HEROES
        print(f"\n{'=' * 70}")
        print(f"     ОПТИМИЗАЦИЯ ДНЯ 1 С {heroes} ГЕРОЯМИ")
        print(f"{'=' * 70}")


# =================================================================================
# БЛОК 10: УТИЛИТЫ ДЛЯ ВЫВОДА
# =================================================================================

class PrettyPrinter:
    """
    Красивое форматирование вывода результатов.
    """

    @staticmethod
    def print_title(text: str, width: int = 80, char: str = '='):
        """Печатает заголовок."""
        print(f"\n{char * width}")
        print(f"{text:^{width}}")
        print(f"{char * width}")

    @staticmethod
    def print_subtitle(text: str, width: int = 70, char: str = '─'):
        """Печатает подзаголовок."""
        print(f"\n{text}")
        print(f"{char * width}")

    @staticmethod
    def print_result(score: int, reward: int, max_hero: int, objects: int,
                     is_best: bool = False):
        """Печатает строку с результатом."""
        best_marker = " ★ НОВЫЙ ЛУЧШИЙ!" if is_best else ""
        print(f"   Счет: {score:>12,}  |  "
              f"Золото: {reward:>5,}  |  "
              f"Героев: {max_hero:>2}  |  "
              f"Мельниц: {objects:>3}{best_marker}")

    @staticmethod
    def print_final(result: SimulationResult, filename: str):
        """Печатает финальные результаты."""
        width = 70
        print(f"\n{'✨' * (width // 4)}")
        print(f"{' ' * ((width - 30) // 2)}ЛУЧШЕЕ НАЙДЕННОЕ РЕШЕНИЕ")
        print(f"{'✨' * (width // 4)}")
        print(f"{'─' * width}")
        print(f"🏆 ИТОГОВЫЙ СЧЕТ:        {result.score:>19,}")
        print(f"💰 СОБРАННОЕ ЗОЛОТО:     {result.reward:>19,}")
        print(f"👥 ЗАТРАТЫ НА ГЕРОЕВ:    {result.hero_cost:>19,}")
        print(f"📈 МАКСИМАЛЬНЫЙ ГЕРОЙ:   {result.max_hero:>19}")
        print(f"🎯 ПОСЕЩЕННЫХ МЕЛЬНИЦ:   {result.unique_objects:>19}")
        print(f"{'─' * width}")
        print(f"💾 Решение сохранено: {filename}")


# =================================================================================
# БЛОК 11: ТЕСТИРОВАНИЕ КОНФИГУРАЦИЙ
# =================================================================================

class ConfigTester:
    """
    Тестирование различных конфигураций параметров.

    Конфигурации: (количество_героев, время_дня1, время_других_дней)
    """

    CONFIGS = [
        (20, 120, 10), (20, 180, 15), (19, 120, 10),
        (19, 180, 15), (21, 120, 10), (21, 180, 15), (18, 180, 15)
    ]

    def __init__(self):
        self.printer = PrettyPrinter()
        self.best_solution = None
        self.best_result = None

    def run_all(self) -> Tuple[List[Tuple[int, int]], SimulationResult]:
        """
        Запускает все конфигурации и возвращает лучшее решение.
        """
        self.printer.print_title("ЗАПУСК ОПТИМИЗАЦИИ", 70)
        print(f"📋 Тестируется {len(self.CONFIGS)} вариантов параметров")

        for idx, (heroes, t1, tn) in enumerate(self.CONFIGS, 1):
            print(f"\n{idx:2}. Героев: {heroes:2} | "
                  f"День 1: {t1:3}с | Дни 2-7: {tn:2}с")
            print(f"{'─' * 60}")

            try:
                # Оптимизация
                optimizer = HeroRouteOptimizer()
                solution = optimizer.optimize(
                    num_heroes=heroes,
                    time_day1=t1,
                    time_other=tn,
                    verbose=False
                )

                if not solution:
                    print("   ❌ Решение не найдено")
                    continue

                # Локальное улучшение
                solution = optimizer.local_search.improve(solution)

                # Оценка
                result = optimizer.simulator.simulate(solution)
                result.unique_objects = len(set(obj for _, obj in solution))

                # Проверка, лучше ли это решение
                is_best = (self.best_result is None or
                           result.score > self.best_result.score)

                self.printer.print_result(
                    result.score, result.reward,
                    result.max_hero, result.unique_objects,
                    is_best
                )

                if is_best:
                    self.best_solution = solution
                    self.best_result = result
                    # Сохраняем лучшее решение
                    write_solution(solution, FilePaths.SUBMISSION)

            except Exception as e:
                print(f"   ❌ Ошибка: {e}")

        return self.best_solution, self.best_result


# =================================================================================
# БЛОК 12: ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =================================================================================

def write_solution(solution: List[Tuple[int, int]], filepath: str) -> None:
    """
    Записывает решение в CSV файл.

    Args:
        solution: Список пар (hero_id, object_id)
        filepath: Путь для сохранения
    """
    pd.DataFrame(solution, columns=["hero_id", "object_id"]).to_csv(
        filepath, index=False, encoding="utf-8"
    )


# =================================================================================
# БЛОК 13: ТОЧКА ВХОДА
# =================================================================================

def main():
    """
    Главная функция программы.

    Последовательность действий:
    1. Вывод приветствия
    2. Тестирование различных конфигураций
    3. Сохранение лучшего решения
    4. Вывод финальных результатов
    """
    printer = PrettyPrinter()

    # Приветствие
    printer.print_title(" 🚀 ОПТИМИЗАЦИЯ МАРШРУТОВ ГЕРОЕВ 🚀 ", 80, '★')


    # Тестирование конфигураций
    tester = ConfigTester()
    best_solution, best_result = tester.run_all()

    # Финальный вывод
    if best_solution is not None:
        printer.print_final(best_result, FilePaths.SUBMISSION)
    else:
        print("\n❌ Не удалось найти ни одного решения.")
        print("   Попробуйте изменить параметры оптимизации.")

    print("\n✅ Работа программы завершена.")


if __name__ == "__main__":
    main()