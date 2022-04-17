import numpy as np
import random
import pandas as pd
from sklearn.cluster import DBSCAN,KMeans
import scipy.stats
from scipy.stats import moment
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,balanced_accuracy_score,accuracy_score
from tqdm import tqdm
class TSProcessor:
    def __init__(self, points_in_template: int, max_template_spread: int):

        # максимальное расстояние между соседними зубчиками шаблона
        self._max_template_spread = max_template_spread
        self._train_series = pd.DataFrame(columns=['percentile_diff','second_moment','third_moment','forth_moment','entropy','max_min_delta','target'])
        self.x_dim: int = max_template_spread ** (points_in_template - 1)  # сколько у нас всего шаблонов
        self.z_dim: int = points_in_template                               # сколько зубчиков в каждом шаблоне

        # сами шаблоны
        templates = (np.repeat(0, self.x_dim).reshape(-1, 1), )

        # непонятный код, который заполняет шаблоны нужными значениями. Пытаться вникнуть бесполезно.
        for i in range(1, points_in_template):
            col = (np.repeat(
                np.arange(1, max_template_spread + 1, dtype=int), max_template_spread ** (points_in_template - (i + 1))
            ) + templates[i - 1][::max_template_spread ** (points_in_template - i)]).reshape(-1, 1)

            templates += (col, )  # don't touch

        self._templates: np.ndarray = np.hstack(templates)

        # формы шаблонов, т.е. [1, 1, 1], [1, 1, 2] и т.д.
        self._template_shapes: np.ndarray = self._templates[:, 1:] - self._templates[:, :-1]

    def fit(self, time_series: np.ndarray) -> None:
        '''Обучить класс на конкретном ряду.'''

        self._time_series   = time_series
        self.y_dim          = self._time_series.size - self._templates[0][-1]
        self._original_size = self._time_series.size

        # создать обучающее множество
        # Его можно представить как куб, где по оси X идут шаблоны, по оси Y - вектора,
        # а по оси Z - индивидуальные точки векторов.
        # Чтобы получить точку A вектора B шаблона C - делаем self._training_vectors[C, B, A].
        # Вектора идут в хронологическом порядке "протаскивания" конкретного шаблона по ряду,
        # шаблоны - по порядку от [1, 1, ... , 1], [1, 1, ..., 2] до [n, n, ..., n].
        self._training_vectors: np.ndarray = \
            np.full(shape=(self.x_dim, self.y_dim, self.z_dim), fill_value=np.inf, dtype=float)

        # тащим шаблон по ряду
        for i in range(self.x_dim):
            template_data = (
                self._time_series[self._templates[i]
                                  + np.arange(self._time_series.size - self._templates[i][-1])[:, None]]
            )

            self._training_vectors[i, :template_data.shape[0]] = (
                self._time_series[self._templates[i]
                                  + np.arange(self._time_series.size - self._templates[i][-1])[:, None]]
            )
    def _freeze_point_rand_clust(self, points_pool: np.ndarray, how: str, dbs_eps: float = 0.0, dbs_min_samples: int = 0,threshold: float = 0.8) -> float:
        if points_pool.size == 0:
            result = np.nan
        else:
            if how == 'mean':
                result = float(points_pool.mean())

            elif how == 'mf':
                points, counts = np.unique(points_pool, return_counts=True)
                result         = points[counts.argmax()]

            elif how == 'cl':
                dbs = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)
                dbs.fit(points_pool.reshape(-1, 1))

                cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)
                #print(cluster_labels.size)
                if cluster_labels.size > 0:
                    clust_for_choice = cluster_labels[((cluster_sizes / cluster_sizes.max()).round(2) > threshold)]
                    #print(cluster_sizes)
                    lab_rand = np.random.choice(clust_for_choice,1)
                    #print(lab_rand)
                    biggest_cluster_center = points_pool[dbs.labels_ == lab_rand].mean()
                    result                 = biggest_cluster_center
                # else:
                #     points, counts = np.unique(points_pool, return_counts=True)
                #     result         = points[counts.argmax()]

        return result
    def pull_traj_clust(self, steps: int, eps: float, n_trajectories: int, noise_amp: float,threshold: float = 0.8) -> np.ndarray:
        self._training_vectors = np.hstack([self._training_vectors,
                                            np.full([self.x_dim, steps, self.z_dim], fill_value=np.inf)])

        # удлиняем изначальый ряд на значение steps
        self._time_series          = np.resize(self._time_series, self._original_size + steps)
        self._time_series[-steps:] = np.nan

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        forecast_sets = np.full((steps, n_trajectories), np.nan)

        for i in tqdm(range(n_trajectories)):
            for j in range(steps):
                #print(j)
                # тестовые вектора, которые будем сравнивать с тренировочными
                last_vectors = (self._time_series[:self._original_size + j]
                                                 [np.cumsum(-self._template_shapes[:, ::-1], axis=1)[:, ::-1]])

                distance_matrix = self._calc_distance_matrix(last_vectors, np.repeat(True, self.x_dim), steps)

                # последние точки тренировочных векторов, оказавшихся в пределах eps
                points = self._training_vectors[distance_matrix < eps][:, -1]

                # теперь нужно выбрать финальное прогнозное значение из возможных
                # я выбираю самое часто встречающееся значение, но тут уже можно на свое усмотрение
                forecast_point                             = self._freeze_point_rand_clust(points, 'cl',dbs_eps=0.001,threshold=threshold)
                forecast_sets[j, i]                        = forecast_point
                self._time_series[self._original_size + j] = forecast_point

                # у нас появилась новая точка в ряду, последние вектора обновились, добавим их в обучающие
                new_training_vectors = (
                    self._time_series[:self._original_size + 1 + j]
                    [np.hstack((np.cumsum(-self._template_shapes[:, ::-1], axis=1)[:, ::-1]
                     - 1, np.repeat(-1, self.x_dim).reshape(-1, 1)))]
                )

                self._training_vectors[:, self.y_dim + j, :] = new_training_vectors

            # честно говоря, я не помню, зачем нужен код дальше

            # delete added vectors after each run
            self._training_vectors[:, self.y_dim:] = np.inf

            # delete added points after each run
            self._time_series[-steps:] = np.nan

        return forecast_sets

    def pull(self, steps: int, eps: float, n_trajectories: int, noise_amp: float) -> np.ndarray:
        '''
        Основной метод пулла, который использовался в статье.

        Parameters
        ----------
        steps : int
            На сколько шагов прогнозируем.
        eps : float
            Минимальное Евклидово расстояние от соответствующего шаблона, в пределах которого должны находиться
            вектора наблюдений, чтобы считаться "достаточно похожими".
        n_trajectories : int
            Сколько траекторий использовать. Чем больше, тем дольше время работы и потенциально точнее результат.
        noise_amp : float
            Максимальная амплитуда шума, используемая при расчете траекторий.

        Возвращает матрицу размером steps x n_trajectories, где по горизонтали идут шаги, а по вертикали - прогнозы
        каждой из траекторий на этом шаге.
        '''

        # прибавляем к тренировочному датасету steps пустых векторов, которые будем заполнять значениями на ходу
        self._training_vectors = np.hstack([self._training_vectors,
                                            np.full([self.x_dim, steps, self.z_dim], fill_value=np.inf)])

        # удлиняем изначальый ряд на значение steps
        self._time_series          = np.resize(self._time_series, self._original_size + steps)
        self._time_series[-steps:] = np.nan

        # сеты прогнозных значений для каждой точки, в которой будем делать прогноз
        forecast_sets = np.full((steps, n_trajectories), np.nan)

        for i in tqdm(range(n_trajectories)):
            for j in range(steps):

                # тестовые вектора, которые будем сравнивать с тренировочными
                last_vectors = (self._time_series[:self._original_size + j]
                                                 [np.cumsum(-self._template_shapes[:, ::-1], axis=1)[:, ::-1]])

                distance_matrix = self._calc_distance_matrix(last_vectors, np.repeat(True, self.x_dim), steps)

                # последние точки тренировочных векторов, оказавшихся в пределах eps
                points = self._training_vectors[distance_matrix < eps][:, -1]

                # теперь нужно выбрать финальное прогнозное значение из возможных
                # я выбираю самое часто встречающееся значение, но тут уже можно на свое усмотрение
                forecast_point                             = self._freeze_point(points, 'mf') \
                    + random.uniform(-noise_amp, noise_amp)
                forecast_sets[j, i]                        = forecast_point
                self._time_series[self._original_size + j] = forecast_point

                # у нас появилась новая точка в ряду, последние вектора обновились, добавим их в обучающие
                new_training_vectors = (
                    self._time_series[:self._original_size + 1 + j]
                    [np.hstack((np.cumsum(-self._template_shapes[:, ::-1], axis=1)[:, ::-1]
                     - 1, np.repeat(-1, self.x_dim).reshape(-1, 1)))]
                )

                self._training_vectors[:, self.y_dim + j, :] = new_training_vectors

            # честно говоря, я не помню, зачем нужен код дальше

            # delete added vectors after each run
            self._training_vectors[:, self.y_dim:] = np.inf

            # delete added points after each run
            self._time_series[-steps:] = np.nan

        return forecast_sets

    def cluster_sets(self, forecast_sets: np.ndarray, dbs_eps: float, dbs_min_samples: int,smooth: bool = False):
        '''
        Скластеризировать полученные в результате пулла множества прогнозных значений.
        Возвращает центр самого большого кластера на каждом шаге.
        '''

        predictions = np.full(shape=[forecast_sets.shape[0], ], fill_value=np.nan)
        dbs         = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)
        aver_spread = []
        for i in range(len(forecast_sets)):
            curr_set = forecast_sets[i]
            dbs.fit(curr_set.reshape(-1, 1))

            cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)

            if cluster_labels.size > 0:
                biggest_cluster_center = curr_set[dbs.labels_ == cluster_labels[cluster_sizes.argmax()]].mean()
                predictions[i] = biggest_cluster_center
                aver_spread.append(max(curr_set) - min(curr_set))
            elif smooth == True:
              if len(aver_spread)!= 0 and max(curr_set) - min(curr_set) < 3*np.mean(aver_spread):
                   predictions[i] = curr_set.mean()
        return predictions

    def _entropy(self,data):
        pd_series = pd.Series(data)
        counts = pd_series.value_counts()
        entropy = scipy.stats.entropy(counts)
        return entropy
    def _detected_class(self,curr_set,dbs_eps,dbs_min_samples):
        dbs         = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)
        dbs.fit(curr_set.reshape(-1, 1))

        cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)

        if cluster_labels.size > 0:
            biggest_cluster_center = curr_set[dbs.labels_ == cluster_labels[cluster_sizes.argmax()]].mean()
            pred = biggest_cluster_center
        else:
            pred = curr_set.mean()
        return pred
    def _make_data_frame(self,forecast_sets: np.ndarray,true_value: np.ndarray,eps: float,dbs_eps: float, dbs_min_samples: int):
        data = []
        for i in range(len(forecast_sets)):
            class_ = 0
            det_class_ = self._detected_class(forecast_sets[i],dbs_eps,dbs_min_samples)
            if ( det_class_ >= true_value[i] - eps and det_class_ <= true_value[i] + eps):
                class_ = 1
            data.append([np.percentile(forecast_sets[i], 75) - np.percentile(forecast_sets[i], 25),\
                        moment(forecast_sets[i],2),moment(forecast_sets[i],3),moment(forecast_sets[i],4),\
                        self._entropy(forecast_sets[i]),np.max(forecast_sets[i]) - np.min(forecast_sets[i]),class_])
        self._train_series = self._train_series.append(pd.DataFrame(data,columns=['percentile_diff','second_moment','third_moment','forth_moment','entropy','max_min_delta','target']),ignore_index=True)
        #pd_series = pd.DataFrame(data,columns=['percentile_diff','second_moment','third_moment','forth_moment','entropy','target'])
        #return pd_series
    def clean_train_set(self):
        self._train_series = pd.DataFrame(columns=['percentile_diff','second_moment','third_moment','forth_moment','entropy','max_min_delta','target'])
    def set_train_set(self,dfs):
        self._train_series = dfs
    def get_train_set(self):
        return self._train_series
    def accumulate_train_set(self, forecast_sets: np.ndarray,true_value: np.ndarray,eps: float,dbs_eps: float, dbs_min_samples: int):
        self._make_data_frame(forecast_sets,true_value,eps,dbs_eps,dbs_min_samples)
        #return self._train_series
    def set_best_estimator(self,estim):
        self._best_estimator = estim
    def learning_by_acc_set(self):
        self._estimator = RandomForestClassifier(random_state=13,\
                          bootstrap=True,\
                          max_depth=80,\
                          max_features=3,\
                          min_samples_leaf=4,\
                          min_samples_split=8,\
                          n_estimators=1000)
        y_train = self._train_series['target']
        x_train = self._train_series.drop(columns=['target'],inplace=False)
        self._estimator.fit(x_train,y_train)

    def learning_by_acc_set_gridCV(self,bootstrap_list:list = [True],\
                                        max_depth_list:list = [1,2,3,4,5,6,7,8,9,10,20,100],\
                                        max_features_list:list = [1,2,3,4],\
                                        min_samples_leaf_list:list = [1,2,3,4,5,6,7,8,9,10,20,100],\
                                        min_samples_split_list:list = [1,2,3,4,5,6,7,8,9,10,20,100],
                                        n_estimators_list:list = [100,200,1000],
                                        steps:int = 100,
                                        kmeans_cl:int = 30):
        train_df = self._train_series
        train_df['target'] = train_df['target'].astype(float)
        y = train_df['target']
        X = train_df.drop(columns=['target'],inplace=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=13)
        k_means =  KMeans(n_clusters=kmeans_cl)
        k_means = k_means.fit(X_train)
        clusters_train = k_means.predict(X_train)
        print(f'KMeans accuracy_score = {accuracy_score(y_train,clusters_train)}')
        k_means =  KMeans(n_clusters=kmeans_cl)
        k_means = k_means.fit(X_test)
        clusters_test = k_means.predict(X_test)
        print(f'KMeans accuracy_score = {accuracy_score(y_test,clusters_test)}')

        X_train_ready = pd.concat([X_train.reset_index(),pd.DataFrame(clusters_train,columns=['clusters']).reset_index()], axis=1)
        X_train_ready.columns = ['step', 'percentile_diff', 'second_moment', 'third_moment',\
               'forth_moment', 'entropy', 'max_min_delta', 'index', 'clusters']
        X_train_ready.drop(columns=['index'],inplace=True)
        X_train_ready['step'] = X_train_ready['step']%steps

        X_test_ready = pd.concat([X_test.reset_index(),pd.DataFrame(clusters_test,columns=['clusters']).reset_index()], axis=1)
        X_test_ready.columns = ['step', 'percentile_diff', 'second_moment', 'third_moment',\
               'forth_moment', 'entropy', 'max_min_delta', 'index', 'clusters']
        X_test_ready.drop(columns=['index'],inplace=True)
        X_test_ready['step'] = X_test_ready['step']%steps
        
        param_grid = {
            'bootstrap': bootstrap_list,
            'max_depth': max_depth_list,
            'max_features': max_features_list,
            'min_samples_leaf': min_samples_leaf_list,
            'min_samples_split': min_samples_split_list,
            'n_estimators': n_estimators_list
        }
        rf = RandomForestClassifier(random_state=13)
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
        grid_search.fit(X_train_ready, y_train)
        print(f'Best params for RandomForestClassifier {grid_search.best_params_}')
        y_test_pred = grid_search.best_estimator_.predict(X_test_ready)
        print(f'accuracy_score = {accuracy_score(y_test,y_test_pred)}')
        self.set_best_estimator(grid_search.best_estimator_)


    def decider_by_forecast(self, forecast_sets: np.ndarray,dbs_eps: float, dbs_min_samples: int,steps: int):
        data = []
        predictions = np.full(shape=[forecast_sets.shape[0], ], fill_value=np.nan)
        for i in range(len(forecast_sets)):
            data.append([np.percentile(forecast_sets[i], 75) - np.percentile(forecast_sets[i], 25),\
                        moment(forecast_sets[i],2),moment(forecast_sets[i],3),moment(forecast_sets[i],4),\
                        self._entropy(forecast_sets[i]),np.max(forecast_sets[i]) - np.min(forecast_sets[i])])
        pd_series = pd.DataFrame(data,columns=['percentile_diff','second_moment','third_moment','forth_moment','entropy','max_min_delta'])
        k_means =  KMeans(n_clusters=30)
        k_means = k_means.fit(pd_series)
        clusters_test = k_means.predict(pd_series)
        X_test_ready = pd.concat([pd_series.reset_index(),pd.DataFrame(clusters_test,columns=['clusters']).reset_index()], axis=1)
        X_test_ready.columns = ['step', 'percentile_diff', 'second_moment', 'third_moment',\
               'forth_moment', 'entropy', 'max_min_delta', 'index', 'clusters']
        X_test_ready.drop(columns=['index'],inplace=True)
        X_test_ready['step'] = X_test_ready['step']%steps
        pred = self._best_estimator.predict(X_test_ready)
        print(f'Number of predicted points = {sum(pred)}')
        for i in range(len(forecast_sets)):
            if pred[i] == 1:
                predictions[i] = self._detected_class(forecast_sets[i],dbs_eps,dbs_min_samples)
        return predictions
        
    def _calc_distance_matrix(self, test_vectors: np.ndarray, mask: np.ndarray, steps: int) -> np.ndarray:
        '''
        По необъяснимым причинам считать матрицу расстояний между тестовыми векторами и тренировочными быстрее вот так.
        '''

        distance_matrix = ((self._training_vectors[mask, :, 0] - np.repeat(test_vectors[:, 0], self.y_dim + steps)
                            .reshape(-1, self.y_dim + steps)) ** 2
                           + (self._training_vectors[mask, :, 1] - np.repeat(test_vectors[:, 1], self.y_dim + steps)
                              .reshape(-1, self.y_dim + steps)) ** 2
                           + (self._training_vectors[mask, :, 2] - np.repeat(test_vectors[:, 2], self.y_dim + steps)
                              .reshape(-1, self.y_dim + steps)) ** 2) ** 0.5

        return distance_matrix

    def _freeze_point(self, points_pool: np.ndarray, how: str, dbs_eps: float = 0.0, dbs_min_samples: int = 0) -> float:
        '''
        Выбрать финальный прогноз в данной точке из множества прогнозных значений.

        "How" варианты:
            "mean" = "mean"
            "mf"   = "most frequent"
            "cl"   = "cluster", нужны dbs_eps и dbs_min_samples
        '''

        if points_pool.size == 0:
            result = np.nan
        else:
            if how == 'mean':
                result = float(points_pool.mean())

            elif how == 'mf':
                points, counts = np.unique(points_pool, return_counts=True)
                result         = points[counts.argmax()]

            elif how == 'cl':
                dbs = DBSCAN(eps=dbs_eps, min_samples=dbs_min_samples)
                dbs.fit(points_pool.reshape(-1, 1))

                cluster_labels, cluster_sizes = np.unique(dbs.labels_[dbs.labels_ > -1], return_counts=True)

                if (cluster_labels.size > 0
                        and np.count_nonzero(((cluster_sizes / cluster_sizes.max()).round(2) > 0.8)) == 1):
                    biggest_cluster_center = points_pool[dbs.labels_ == cluster_labels[cluster_sizes.argmax()]].mean()
                    result                 = biggest_cluster_center
                else:
                    result = np.nan

        return result
