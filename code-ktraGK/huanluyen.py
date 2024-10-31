
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, x):
        if self.metric == 'euclidean':
            distances = np.linalg.norm(self.X_train - x, axis=1)
        else:
            raise ValueError(
                "Currently, only 'euclidean' metric is supported.")
        return distances

    def predict(self, X):
        predictions = []
        for x in X:
            distances = self._compute_distances(x)
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]

            if self.weights == 'uniform':
                most_common = np.bincount(nearest_labels).argmax()
            elif self.weights == 'distance':
                # Avoid division by zero
                weights = 1 / (distances[nearest_indices] + 1e-5)
                weighted_votes = np.zeros(len(np.unique(self.y_train)))
                for i, label in enumerate(nearest_labels):
                    weighted_votes[label] += weights[i]
                most_common = np.argmax(weighted_votes)
            else:
                raise ValueError(
                    "Weights must be either 'uniform' or 'distance'.")

            predictions.append(most_common)
        return np.array(predictions)


class LinearRegression:
    def fit(self, X, y):
        # Thêm cột bias vào X (cột các giá trị 1 để tính toán hệ số tự do)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # thêm cột 1 cho hệ số tự do
        # Tính theta theo công thức bình phương tối thiểu
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        # Thêm cột bias cho X để dự đoán
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        y = y.to_numpy()  # Convert y to a NumPy array
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if n_samples >= self.min_samples_split and (self.max_depth is None or depth < self.max_depth):
            best_split = self._best_split(X, y, n_features)
            if best_split is not None:
                left_indices, right_indices = best_split['indices']
                left_child = self._grow_tree(
                    X[left_indices], y[left_indices], depth + 1)
                right_child = self._grow_tree(
                    X[right_indices], y[right_indices], depth + 1)
                return self.Node(feature=best_split['feature'], threshold=best_split['threshold'],
                                 left=left_child, right=right_child)

        # Leaf node
        leaf_value = np.mean(y)
        return self.Node(value=leaf_value)

    def _best_split(self, X, y, n_features):
        best_split = {}
        min_mse = float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)[0]
                right_indices = np.where(X[:, feature] > threshold)[0]

                if len(left_indices) >= self.min_samples_leaf and len(right_indices) >= self.min_samples_leaf:
                    mse = self._calculate_mse(
                        y[left_indices], y[right_indices])
                    if mse < min_mse:
                        min_mse = mse
                        best_split = {
                            'feature': feature,
                            'threshold': threshold,
                            'indices': (left_indices, right_indices)
                        }

        return best_split if best_split else None

    def _calculate_mse(self, left, right):
        left_mean = np.mean(left) if len(left) > 0 else 0
        right_mean = np.mean(right) if len(right) > 0 else 0
        mse = (np.sum((left - left_mean) ** 2) +
               np.sum((right - right_mean) ** 2)) / (len(left) + len(right))
        return mse

    def predict(self, X):
        return np.array([self._predict_single(sample, self.tree) for sample in X])

    def _predict_single(self, x, node):
        if node.value is not None:  # Leaf node
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)


class DecisionTreeClassifier:
    def __init__(self,
                 # Tiêu chí tách node ('gini' hoặc 'entropy')
                 criterion="gini",
                 # Chiến lược chia tách ('best' hoặc 'random')
                 splitter="best",
                 max_depth=None,  # Độ sâu tối đa của cây
                 min_samples_split=2,  # Số lượng mẫu tối thiểu để chia tách một node
                 min_samples_leaf=1,  # Số lượng mẫu tối thiểu cho một node lá
                 max_features=None,  # Số lượng feature tối đa để xét khi chia tách
                 random_state=None,  # Seed để tái tạo kết quả
                 max_leaf_nodes=None,  # Số lượng node lá tối đa
                 min_impurity_decrease=0.0):  # Ngưỡng giảm impurity tối thiểu để chia tách node
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.tree_ = None
        if random_state:
            np.random.seed(random_state)

    class Node:
        def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
            self.gini = gini  # Chỉ số Gini của node
            self.num_samples = num_samples  # Số lượng mẫu tại node
            self.num_samples_per_class = num_samples_per_class  # Số lượng mẫu mỗi lớp
            self.predicted_class = predicted_class  # Lớp dự đoán tại node
            self.feature_index = 0  # Chỉ số feature dùng để tách node
            self.threshold = 0  # Ngưỡng phân tách
            self.left = None  # Nhánh trái
            self.right = None  # Nhánh phải

    def _gini(self, y):
        """Tính chỉ số Gini."""
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _entropy(self, y):
        """Tính entropy."""
        m = len(y)
        entropy = 0
        for c in np.unique(y):
            p = np.sum(y == c) / m
            entropy -= p * np.log2(p)
        return entropy

    def _impurity(self, y):
        """Tính độ giảm impurity theo criterion."""
        if self.criterion == "gini":
            return self._gini(y)
        elif self.criterion == "entropy":
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _best_split(self, X, y):
        """Tìm ra điểm chia tách tốt nhất."""
        m, n = X.shape
        if m <= 1:
            return None, None

        # Ánh xạ lớp thành chỉ số
        unique_classes = np.unique(y)
        num_parent = [np.sum(y == c) for c in unique_classes]
        best_gini = self._impurity(y)
        best_idx, best_thr = None, None

        # Tính toán số lượng features sẽ xét
        if self.max_features is None:
            features = range(n)
        else:
            features = np.random.choice(n, self.max_features, replace=False)

        # Thực hiện chia tách trên từng feature
        for idx in features:
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * len(unique_classes)
            num_right = num_parent.copy()

            for i in range(1, m):
                # Ánh xạ lớp thành chỉ số
                c = np.where(unique_classes == classes[i - 1])[0][0]
                num_left[c] += 1
                num_right[c] -= 1

                gini_left = 1.0 - \
                    sum((num_left[x] / i) **
                        2 for x in range(len(unique_classes)))
                gini_right = 1.0 - \
                    sum((num_right[x] / (m - i)) **
                        2 for x in range(len(unique_classes)))

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        """Xây dựng cây đệ quy."""
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = self.Node(
            gini=self._impurity(y),
            num_samples=y.shape[0],
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Điều kiện dừng
        if (self.max_depth is None or depth < self.max_depth) and node.num_samples >= self.min_samples_split and node.gini > 0:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def fit(self, X, y):
        """Huấn luyện cây quyết định."""
        self.tree_ = self._grow_tree(X, y)

    def _predict(self, inputs, node):
        """Dự đoán cho một input dựa trên cây."""
        if node.left is None and node.right is None:
            return node.predicted_class
        if inputs[node.feature_index] < node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)

    def predict(self, X):
        """Dự đoán cho các input."""
        return [self._predict(inputs, self.tree_) for inputs in X]


class LogisticRegression:
    def __init__(self, penalty="l2", C=1.0, fit_intercept=True, max_iter=100, tol=1e-4, random_state=None):
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.weights = None
        self.intercept = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y == 0, -1, 1)  # Chuyển đổi nhãn

        # Khởi tạo weights
        self.weights = np.zeros(n_features)
        self.intercept = 0

        # Huấn luyện mô hình
        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.intercept
            y_pred = self.sigmoid(linear_model)

            # Tính gradient
            error = y_pred - (y + 1) / 2
            gradient = np.dot(X.T, error) / n_samples

            # Penalize
            if self.penalty == "l2":
                gradient += (1 / self.C) * self.weights  # L2 regularization
            elif self.penalty == "l1":
                gradient += (1 / self.C) * \
                    np.sign(self.weights)  # L1 regularization

            # Cập nhật weights
            self.weights -= self.tol * gradient

            # Cập nhật intercept
            self.intercept -= self.tol * np.mean(error)

            # Kiểm tra hội tụ
            if np.linalg.norm(gradient) < self.tol:
                break

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.intercept
        y_pred = self.sigmoid(linear_model)
        return np.where(y_pred >= 0.5, 1, 0)


class SVC:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3,
                 coef0=0.0, tol=1e-3, max_iter=1000, random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None

        # Thiết lập ngẫu nhiên nếu random_state được cung cấp
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y == 0, -1, 1)  # Chuyển đổi nhãn từ {0, 1} sang {-1, 1}

        # Khởi tạo alpha
        self.alpha = np.zeros(n_samples)
        self.b = 0.0

        # Khai báo kernel
        if self.kernel == 'rbf':
            K = self.rbf_kernel(X, X)
        elif self.kernel == 'linear':
            K = self.linear_kernel(X, X)
        elif self.kernel == 'poly':
            K = self.poly_kernel(X, X)
        else:
            raise ValueError("Unsupported kernel")

        # Huấn luyện SVM
        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)

            # Chọn ngẫu nhiên một chỉ số cho việc cập nhật
            random_indices = np.random.permutation(n_samples)

            for i in random_indices:
                # Tính giá trị dự đoán
                decision = np.dot(self.alpha * y_, K[:, i]) + self.b

                # Điều kiện KKT
                if (y_[i] * decision < 1):
                    # Cập nhật alpha
                    self.alpha[i] += self.C * (1 - y_[i] * decision)
                    self.b += self.C * y_[i]

                # Giảm alpha
                self.alpha[i] *= (1 - 1 / (2 * self.C))

            # Nếu không có thay đổi nào trong alpha, dừng huấn luyện
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break

        # Lưu trữ các vector hỗ trợ
        self.support_vectors = X[self.alpha > 0]
        self.support_vector_labels = y_[self.alpha > 0]

    def rbf_kernel(self, X1, X2):
        if self.gamma == 'scale':
            gamma_value = 1.0 / (X1.shape[1] * np.var(X1))
        else:
            gamma_value = self.gamma
        return np.exp(-gamma_value * (np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2))

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def poly_kernel(self, X1, X2):
        return (np.dot(X1, X2.T) + self.coef0) ** self.degree

    def predict(self, X):
        if self.kernel == 'rbf':
            K = self.rbf_kernel(X, self.support_vectors)
        elif self.kernel == 'linear':
            K = self.linear_kernel(X, self.support_vectors)
        elif self.kernel == 'poly':
            K = self.poly_kernel(X, self.support_vectors)

        decision = np.dot(
            K, self.alpha[self.alpha > 0] * self.support_vector_labels) + self.b
        return np.where(decision >= 0, 1, 0)


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = {}
        self.classes_ = []

    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        total_count = len(y)

        self.class_priors = {c: count / total_count for c,
                             count in zip(classes, counts)}
        self.classes_ = classes

        feature_counts = {c: np.zeros(X.shape[1]) for c in classes}

        for i in range(len(y)):
            feature_counts[y.iloc[i]] += X[i].toarray()[0]

        for c in classes:
            total_words = feature_counts[c].sum()
            self.feature_probs[c] = (
                feature_counts[c] + self.alpha) / (total_words + self.alpha * len(feature_counts))

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            log_probs = {}
            for c in self.class_priors:
                log_probs[c] = np.log(
                    self.class_priors[c]) + np.sum(np.log(self.feature_probs[c]) * X[i].toarray()[0])
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):  # Updated to n_estimators
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y.iloc[indices]  # Keep y as a Pandas Series

            # Select features for the tree
            if self.max_features == 'auto':
                max_features = int(np.sqrt(n_features))
            elif self.max_features == 'sqrt':
                max_features = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                max_features = int(np.log2(n_features))
            else:
                max_features = self.max_features

            features_indices = np.random.choice(
                n_features, max_features, replace=False)
            # Select only the features for this tree
            X_sample = X_sample[:, features_indices]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, features_indices))

    def predict(self, X):
        # Aggregate predictions from all trees
        tree_preds = np.array([tree.predict(X[:, features])
                              for tree, features in self.trees])
        return np.mean(tree_preds, axis=0)


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators=100,  # Số lượng cây trong rừng
                 # Tiêu chuẩn để đánh giá sự phân tách ('gini' hoặc 'entropy')
                 criterion="gini",
                 max_depth=None,  # Độ sâu tối đa của mỗi cây
                 min_samples_split=2,  # Số lượng mẫu tối thiểu để phân tách một node
                 min_samples_leaf=1,  # Số lượng mẫu tối thiểu cho một node lá
                 max_features="sqrt",  # Số lượng feature tối đa được xét tại mỗi node
                 bootstrap=True,  # Có sử dụng bootstrap cho các mẫu không
                 n_jobs=None,  # Số lượng job chạy song song
                 random_state=None,  # Seed cho việc tái lặp
                 max_samples=None):  # Số lượng mẫu tối đa khi bootstrap

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_samples = max_samples
        self.trees_ = []  # Nơi chứa các cây đã huấn luyện
        if self.random_state:
            np.random.seed(self.random_state)

    def _bootstrap_sample(self, X, y):
        # Tạo một mẫu bootstrap (sampling có lặp lại)
        n_samples = X.shape[0]
        if self.max_samples:
            n_samples = min(n_samples, self.max_samples)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        # Chuyển y thành numpy array nếu là Series của Pandas
        if isinstance(y, pd.Series):
            y = y.to_numpy()

        return X[indices], y[indices]

    def _get_max_features(self, n_features):
        # Tính số lượng features được sử dụng tại mỗi node
        if isinstance(self.max_features, int):
            return self.max_features
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            return n_features  # Sử dụng tất cả features

    def fit(self, X, y):
        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            # Tạo một cây quyết định
            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self._get_max_features(n_features),
                random_state=self.random_state
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)

    def predict(self, X):
        # Thu thập dự đoán từ tất cả các cây
        tree_preds = np.array([tree.predict(X) for tree in self.trees_])
        # Áp dụng voting (đa số thắng)
        y_pred = [Counter(tree_pred).most_common(1)[0][0]
                  for tree_pred in tree_preds.T]
        return np.array(y_pred)

    def predict_proba(self, X):
        # Thu thập xác suất từ tất cả các cây
        tree_probas = np.array([tree.predict_proba(X) for tree in self.trees_])
        # Trung bình xác suất từ tất cả các cây
        avg_proba = np.mean(tree_probas, axis=0)
        return avg_proba


class LassoRegression:
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        if self.normalize:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        for iteration in range(self.max_iter):
            y_pred = np.dot(X, self.coef_)
            residuals = y_pred - y

            # Tính gradient với hình phạt L1
            gradient = (X.T.dot(residuals) / n_samples) + \
                self.alpha * np.sign(self.coef_)
            new_coef = self.coef_ - gradient

            # Kiểm tra điều kiện dừng
            if np.linalg.norm(new_coef - self.coef_) < self.tol:
                break

            self.coef_ = new_coef

        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]  # Trả về hệ số mà không có intercept

    def predict(self, X):
        if self.fit_intercept:
            return np.dot(X, self.coef_) + self.intercept_
        return np.dot(X, self.coef_)


class RidgeRegression:
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        if self.normalize:
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        for iteration in range(self.max_iter):
            y_pred = np.dot(X, self.coef_)
            residuals = y_pred - y

            # Tính gradient với hình phạt L2
            gradient = (X.T.dot(residuals) / n_samples) + \
                self.alpha * self.coef_
            new_coef = self.coef_ - gradient

            # Kiểm tra điều kiện dừng
            if np.linalg.norm(new_coef - self.coef_) < self.tol:
                break

            self.coef_ = new_coef

        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]  # Trả về hệ số mà không có intercept

    def predict(self, X):
        if self.fit_intercept:
            return np.dot(X, self.coef_) + self.intercept_
        return np.dot(X, self.coef_)
