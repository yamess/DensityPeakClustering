def parallele(func, arg, max_worker=4):
    """
    Cett fonction permet de paralleliser les taches de la fonction f
    sur les arguments arg données
    :param func: Fonction de tache
    :param arg: les arguments de la fonction f à évaluer
    :param max_worker: Nombre total de thread à lançer
    :return: Liste des resultats de l'exécution de la fonction f sur pour chaque argument de arg
    """
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_worker=max_worker) as executor:
        result = executor.map(func, arg, timeout=None)
    return result
