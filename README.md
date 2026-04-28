# Titanic — Kaggle Solution

Предсказание выживаемости пассажиров Титаника с помощью нейронной сети на PyTorch.

## Подход

Бинарная классификация через полносвязную сеть с сигмоидой на выходе.

**Признаки:** `Pclass`, `Sex`, `Age`, `Parch`  
Пропуски в `Age` заполняются средним по классу билета. `Sex` кодируется в 0/1.  
Все признаки нормализуются через `StandardScaler`.

## Архитектура

```
Linear(4 → 16) → ReLU → Linear(16 → 32) → ReLU → Linear(32 → 1) → Sigmoid
```

- Loss: `BCELoss`
- Optimizer: `Adam`, lr=0.001
- Epochs: 500
- Train/Val split: 445 / 445

## Запуск

```bash
pip install torch pandas scikit-learn numpy
python train.py
```

На выходе генерируется `answer.csv` с колонками `PassengerId` и `Survived`.

## Файлы

| Файл | Описание |
|------|----------|
| `train.py` | Обучение и генерация предсказаний |
| `newtrain2.csv` | Предобработанный обучающий датасет |
| `test.csv` | Тестовый датасет Kaggle |
| `answer.csv` | Результат для сабмита |
