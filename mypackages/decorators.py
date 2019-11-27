def timing(func):
    """
    Cette fonction est un decorateur qui met de calculer le temps d'execution d'une fonction
    :param func: la fonction a évaluer
    :return: exécute la fonction et le temps d'exécution
    """
    import time

    def wrapper(*args, **kwargs):
        print("Calculation start")
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution time: {round(end_time-start, 2)} second(s)")
        print("Finished!")
        return res
    return wrapper
