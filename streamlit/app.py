import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats
import pandas as pd

def laplas_value(a):
    return scipy.stats.norm.ppf((1 - 2 * a) / 2 + 0.5)

def preprocessing_csv(uploader):
    data = pd.read_csv(uploader, encoding='cp1251', quoting=3)
    data.columns = ['days', 'age', 'gender']
    data['days'] = data['days'].apply(lambda x: x.replace('"', ''))
    data['gender'] = data['gender'].apply(lambda x: x.replace('"', ''))
    data['gender'] = data['gender'].replace({'Ж':'f', 'М': 'm'})
    data['days'] = data['days'].astype('int64')
    return data

def u_criteria(df1, df2):
    m_1 = df1.query('is_more').shape[0]
    m_2 = df2.query('is_more').shape[0]
    n_1 = df1.shape[0]
    n_2 = df2.shape[0]
    return ((m_1 / n_1 - m_2 / n_2) /
            np.sqrt((m_1 + m_2) / (n_1 + n_2) * (1 - (m_1 + m_2) / (n_1 + n_2)) * (1 / n_1 + 1 / n_2)))

def main():
    st.header("Дашборд с проверкой гипотез")
    loader = st.file_uploader("Загрузите файл в формате csv", type={"csv"})
    if loader is not None:
        data = preprocessing_csv(loader)
        st.markdown('### Первая гипотеза')
        days_1 = st.slider("Укажите параметр work_days для проверки первой гипотезы: ", min_value=min(data['days']),   
                           max_value=max(data['days']), value=2, step=1)
        data['is_more'] = data['days'] > days_1
        group_m = data.query('gender == "m"')
        group_f = data.query('gender == "f"')
        
        st.markdown("В качестве нулевой гипотезы выдвинем следующую гипотезу:")
        st.markdown(f"* Мужчины и женщины пропускают более {days_1} рабочих дней одинаково часто ($H_0: p_1 = p_2$),")
        st.markdown("а в качестве альтернативной:")
        st.markdown(f"* Мужчины пропускают в течение года более {days_1} рабочих дней значимо чаще женщин ($H_2: p_1 > p_2$)""")
        
        fig, ax = plt.subplots()
        ax.hist(group_m['days'], bins=9)
        plt.title('Распределение количества пропущенных дней среди мужчин', {'fontsize': 11} )
        plt.ylabel('Кол-во сотрудников')
        plt.xlabel('Кол-во дней')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.hist(group_f['days'], bins=9)
        plt.title('Распределение количества пропущенных дней среди женщин', {'fontsize': 11} )
        plt.ylabel('Кол-во сотрудников')
        plt.xlabel('Кол-во дней')
        st.pyplot(fig)
        
        st.markdown("""Проверку гипотез будем осуществлять с помощью метода сравнения вероятности двух биномиальных распределений.
                    Более подробно метод расписан в Jupyter Notebook. Тут же, взглянем непосредственно на результат проверки гипотез""")
        
        alpha_1 = float(st.text_input('Введите уровень значимости в интервале от 0 до 1', value=0.05, key=1))
        res = u_criteria(group_m, group_f) < laplas_value(alpha_1)

        if res:
            st.text('Принимаем нулевую гипотезу')
        else:
            st.text('Отклоняем нулевую гипотезу и принимаем альтернативную')
        
        # вторая гипотеза
        st.markdown('### Вторая гипотеза')
        days_2 = st.slider("Укажите параметр work_days для проверки второй гипотезы: ", min_value=min(data['days']),   
                           max_value=max(data['days']), value=2, step=1)
        data['is_more'] = data['days'] > days_2

        age = st.slider("Укажите параметр age для проверки второй гипотезы: ", min_value=min(data['age']),   
                           max_value=max(data['age']), value=35, step=1)
        group_older = data.query('age >= @age')
        group_younger = data.query('age < @age')
        
        st.markdown("В качестве нулевой гипотезы выдвинем следующую гипотезу:")
        st.markdown(f"* Сотрудники пропускают более {days_2} рабочих дней одинаково часто, независимо от возраста ($H_0: p_1 = p_2$),")
        st.markdown("а в качестве альтернативной:")
        st.markdown(f"* Сотрудники старше {age} пропускают в течение года более {days_2} рабочих дней значимо чаще младших сотрудников ($H_2: p_1 > p_2$)""")
        
        fig, ax = plt.subplots()
        ax.hist(group_older['days'], bins=9)
        plt.title('Распределение количества пропущенных дней среди cтарших сотрудников', {'fontsize': 9} )
        plt.ylabel('Кол-во сотрудников')
        plt.xlabel('Кол-во дней')
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.hist(group_younger['days'], bins=9)
        plt.title('Распределение количества пропущенных дней среди младших сотрудников', {'fontsize': 9} )
        plt.ylabel('Кол-во сотрудников')
        plt.xlabel('Кол-во дней')
        st.pyplot(fig)
        
        alpha_2 = float(st.text_input('Введите уровень значимости в интервале от 0 до 1', value=0.05, key=2))
        res_2 = u_criteria(group_older, group_younger) < laplas_value(alpha_2)

        if res_2:
            st.text('Принимаем нулевую гипотезу')
        else:
            st.text('Отклоняем нулевую гипотезу и принимаем альтернативную')

if __name__ == "__main__":
    main()
