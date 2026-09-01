from typing import final
import pandas as pd
import datetime
import pyodbc
from dateutil.relativedelta import relativedelta
from pathlib import Path

from src.utils.config import load_config


def _id_a_str(series: pd.Series) -> pd.Series:
    """Normaliza una columna identificadora (cédula) a str, sin sufijo '.0'.

    Columnas numéricas de origen externo (Excel, SQL) a veces se tipan como
    float64 (ej. cuando admiten NULL), lo que produce sufijos '.0' al castear
    directo a str (ej. '7601445.0' en vez de '7601445') y rompe silenciosamente
    cualquier merge por Id, sin lanzar ningún error. Pasar primero por Int64
    (nullable) elimina ese sufijo sin perder los NaN reales.

    Args:
        series: Serie con identificadores, en cualquier dtype (str, int64,
            float64).

    Returns:
        pd.Series de dtype str (con NaN reales donde el valor no era numérico
        o era nulo).
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64").astype(str)


def extract_inactivos(
    con_bi:str
) -> pd.DataFrame:
    """
    Consulta en BodegaCorporativa la población de asociados inactivos del mes
    actual (día 1 del mes en curso). Reemplaza al antiguo insumo
    `Poblacion_Inactivos.xlsx`: a diferencia de ese Excel, que podía traer
    varios periodos a la vez, esta consulta siempre trae un único periodo (el
    mes en curso) — por eso agrega la columna 'Periodo' (YYYYMM) al resultado,
    igual para todas las filas, en vez de depender de que el insumo ya la
    traiga.

    Parameters
    ----------
    con_bi : str
        String de conexión pyodbc a BodegaCorporativa.

    Returns
    -------
    pd.DataFrame
        Columnas ID, strcodestadoclientefuente, strDescEstadoFuente, Periodo.

    Raises
    ------
    FileNotFoundError
        Si el archivo poblacion.sql no existe en sql/.
    ValueError
        Si no se obtuvo población para el periodo consultado.
    """
    # --- Se lee la consulta --- #

    query_path = Path.cwd() / "sql" / "poblacion.sql"

    if not query_path.exists():
        raise FileNotFoundError(f'No se encontro el query: {query_path.name} dentro de sql/')

    base_query = query_path.read_text(encoding='utf-8', errors='coerce')

    # --- Periodo de consulta --- #
    
    if datetime.datetime.today().day == 1:
        periodo_dt = (datetime.datetime.today() - relativedelta(months=1)).replace(day=1)
        periodo_ym = (periodo_dt + relativedelta(months=1)).strftime('%Y%m')
    else:
        periodo_dt = datetime.datetime.today().replace(day=1)
        periodo_ym = periodo_dt.strftime('%Y%m')
    
    # --- Se deja un formato fijo --- #
    periodo = periodo_dt.strftime('%Y-%m-%d')
    
    # --- metemos el periodo de consulta --- #
    query_periodo = base_query.replace("'?'", f"'{periodo}'")

    conn_bi = None
    try:
        conn_bi = pyodbc.connect(con_bi)
        print(f'Consultando Inactivos periodo: {periodo}')
        df = pd.read_sql(sql=query_periodo,
                         con=conn_bi, #type:ignore
                         dtype={
                             'ID':int,
                             'strcodestadoclientefuente':str,
                             'strDescEstadoFuente':str,
                         })
        print(f'Consulta realizada con exito✅')
    except Exception as e:
        print(f'Error consultado poblacion: {e}')
        raise
    finally:
        if conn_bi is not None:
            conn_bi.close()

    if len(df) == 0:
        raise ValueError(f'No se obtuvo poblacion para el periodo: {periodo}')

    df['Periodo'] = periodo_ym

    return df

def extract_cantidad_productos(
    inac: pd.DataFrame,
    con_bi: str,
    tamanio_lote: int = 2000,
) -> dict[str, pd.DataFrame]:
    """
    Extrae la cantidad de productos tenidos por cada cédula de `inac`, consultando
    BodegaCorporativa en el periodo inmediatamente anterior al periodo de cada
    cédula (ej. Periodo 202606 en `inac` -> se consulta el periodo 202605).

    La query SQL se obtiene desde sql/cantidad_productos.sql.

    Parameters
    ----------
    inac : pd.DataFrame
        DataFrame de población de inactivos. Debe contener las columnas
        'Periodo' (str YYYYMM) e 'ID'.
    con_bi : str
        String de conexión pyodbc a BodegaCorporativa.
    tamanio_lote : int, optional
        Número máximo de cédulas por lote de consulta. Default 2000.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario {'cantidad_productos': DataFrame} con columnas Periodo, Id,
        Numcantidadproductos.

    Raises
    ------
    FileNotFoundError
        Si el archivo cantidad_productos.sql no existe en sql/.
    ValueError
        Si no se obtuvieron resultados para ningún periodo o lote consultado.
    """
    path_query = Path.cwd() / "sql" / "cantidad_productos.sql"

    if not path_query.exists():
        raise FileNotFoundError(f"La consulta {path_query.name} no existe en sql/")

    base_query = path_query.read_text(encoding="utf-8", errors="coerce")

    df_list = []

    for periodo, grupo in inac.groupby("Periodo"):
        periodo_consulta = (
            datetime.datetime.strptime(f"{periodo}01", "%Y%m%d") - relativedelta(months=1)
        ).strftime("%Y-%m-%d")

        cedulas_str = [str(c) for c in grupo["ID"].unique().tolist()]
        lotes = [
            cedulas_str[i : i + tamanio_lote]
            for i in range(0, len(cedulas_str), tamanio_lote)
        ]

        query_periodo = base_query.replace("'?'", f"'{periodo_consulta}'")
        conn_bi = None
        try:
            conn_bi = pyodbc.connect(con_bi)
            for lot in lotes:
                cedulas_consulta = ",".join(f"'{c}'" for c in lot)
                query_lote = query_periodo.replace("{ids}", cedulas_consulta)
                df_temp = pd.read_sql(sql=query_lote, con=conn_bi)  # type: ignore
                df_list.append(df_temp)
            print(f"Periodo: {periodo_consulta} -- Consultado")
        except Exception as e:
            print(f"Error en periodo {periodo_consulta}: {e}")
            raise
        finally:
            if conn_bi is not None:
                conn_bi.close()

    if not df_list:
        raise ValueError("No se obtuvieron resultados. Verifica la query y los parámetros.")

    return {"cantidad_productos": pd.concat(df_list, axis=0, ignore_index=True)}


def extract_v_360(
    inac: pd.DataFrame,
    con_bi: str,
    tamanio_lote: int = 2000,
) -> dict[str, pd.DataFrame]:
    """
    Extrae de la vista 360 solo las columnas necesarias para el scoring D2-D4
    (Tipo_cliente_*, Valorperseverancia) para las cédulas de `inac`, consultando
    en el periodo inmediatamente anterior al periodo de cada cédula (ej. Periodo
    202606 en `inac` -> se consulta el periodo 202605), igual que
    `extract_cantidad_productos`.

    La query SQL se obtiene desde sql/v_360.sql.

    Parameters
    ----------
    inac : pd.DataFrame
        DataFrame de población de inactivos. Debe contener las columnas
        'Periodo' (str YYYYMM) e 'ID'.
    con_bi : str
        String de conexión pyodbc a BodegaCorporativa.
    tamanio_lote : int, optional
        Número máximo de cédulas por lote de consulta. Default 2000.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario {'v_360': DataFrame} con columnas Periodo, Identificacion,
        Tipo_cliente_bancoomeva, Tipo_cliente_prepagada, Tipo_cliente_seguros,
        Tipo_cliente_adicionales, Valorperseverancia.

    Raises
    ------
    FileNotFoundError
        Si el archivo v_360.sql no existe en sql/.
    ValueError
        Si no se obtuvieron resultados para ningún periodo o lote consultado.
    """
    path_query = Path.cwd() / "sql" / "v_360.sql"

    if not path_query.exists():
        raise FileNotFoundError(f"La consulta {path_query.name} no existe en sql/")

    base_query = path_query.read_text(encoding="utf-8", errors="coerce")

    df_list = []

    for periodo, grupo in inac.groupby("Periodo"):
        periodo_consulta = (
            datetime.datetime.strptime(f"{periodo}01", "%Y%m%d") - relativedelta(months=1)
        ).strftime("%Y-%m-%d")

        cedulas_str = [str(c) for c in grupo["ID"].unique().tolist()]
        lotes = [
            cedulas_str[i : i + tamanio_lote]
            for i in range(0, len(cedulas_str), tamanio_lote)
        ]

        query_periodo = base_query.replace("'?'",f"'{periodo_consulta}'", 1)
        conn_bi = None
        try:
            conn_bi = pyodbc.connect(con_bi)
            for lot in lotes:
                cedulas_consulta = ",".join(f"'{c}'" for c in lot)
                query_lote = query_periodo.replace("{ids}", cedulas_consulta)
                df_temp = pd.read_sql(sql=query_lote, con=conn_bi)  # type: ignore
                df_list.append(df_temp)
            print(f"v_360 -- Periodo: {periodo_consulta} -- Consultado")
        except Exception as e:
            print(f"Error en periodo {periodo_consulta}: {e}")
            raise
        finally:
            if conn_bi is not None:
                conn_bi.close()

    if not df_list:
        raise ValueError("No se obtuvieron resultados. Verifica la query y los parámetros.")

    return {"v_360": pd.concat(df_list, axis=0, ignore_index=True)}


def extract_demografica(
    inac: pd.DataFrame,
    con_bi: str,
    tamanio_lote: int = 2000,
) -> dict[str, pd.DataFrame]:
    """
    Extrae de Demografica solo las columnas necesarias para el scoring D2-D4
    (Ingresos, Cuotas_canceladas_aportes, Saldoaportes, Antiguedad) para las
    cédulas de `inac`, consultando en el periodo inmediatamente anterior al
    periodo de cada cédula, igual que `extract_cantidad_productos`.

    La query SQL se obtiene desde sql/demo.sql.

    Parameters
    ----------
    inac : pd.DataFrame
        DataFrame de población de inactivos. Debe contener las columnas
        'Periodo' (str YYYYMM) e 'ID'.
    con_bi : str
        String de conexión pyodbc a BodegaCorporativa.
    tamanio_lote : int, optional
        Número máximo de cédulas por lote de consulta. Default 2000.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario {'demografica': DataFrame} con columnas Periodo, ID,
        Ingresos, Cuotas_canceladas_aportes, Saldoaportes, Antiguedad.

    Raises
    ------
    FileNotFoundError
        Si el archivo demo.sql no existe en sql/.
    ValueError
        Si no se obtuvieron resultados para ningún periodo o lote consultado.
    """
    path_query = Path.cwd() / "sql" / "demo.sql"

    if not path_query.exists():
        raise FileNotFoundError(f"La consulta {path_query.name} no existe en sql/")

    base_query = path_query.read_text(encoding="utf-8", errors="coerce")

    df_list = []

    for periodo, grupo in inac.groupby("Periodo"):
        periodo_consulta = (
            datetime.datetime.strptime(f"{periodo}01", "%Y%m%d") - relativedelta(months=1)
        ).strftime("%Y-%m-%d")
        
        # --- Periodo final between --- #
        periodo_ym = (
            datetime.datetime.today() - relativedelta(months=1)
        ).strftime("%Y%m")
        
        # --- Periodo inicial between --- #
        periodo_ym_past = (
            datetime.datetime.strptime(periodo_ym, "%Y%m") - relativedelta(months=12)
        ).strftime("%Y%m")
        
        cedulas_str = [str(c) for c in grupo["ID"].unique().tolist()]
        lotes = [
            cedulas_str[i : i + tamanio_lote]
            for i in range(0, len(cedulas_str), tamanio_lote)
        ]
        
        query_periodo = base_query.replace("'?'", f"'{periodo_consulta}'")
        conn_bi = None
        try:
            conn_bi = pyodbc.connect(con_bi)
            for lot in lotes:
                cedulas_consulta = ",".join(f"'{c}'" for c in lot)
                query_lote = query_periodo.replace("{ids}", cedulas_consulta)
                df_temp = pd.read_sql(sql=query_lote, 
                                      con=conn_bi,  # type: ignore
                                      params=[int(periodo_ym_past), 
                                              int(periodo_ym)]) 
                df_list.append(df_temp)
            print(f"demografica -- Periodo: {periodo_consulta} -- Consultado")
        except Exception as e:
            print(f"Error en periodo {periodo_consulta}: {e}")
            raise
        finally:
            if conn_bi is not None:
                conn_bi.close()

    if not df_list:
        raise ValueError("No se obtuvieron resultados. Verifica la query y los parámetros.")

    return {"demografica": pd.concat(df_list, axis=0, ignore_index=True)}

def extract_tenencia_historica(
    inac: pd.DataFrame,
    con_bi: str,
    meses_atras: int = 12,
    tamanio_lote: int = 2000,
) -> dict[str, pd.DataFrame]:
    """
    Extrae el histórico de tenencia de productos (`meses_atras` meses, contados
    hacia atrás desde el periodo de referencia de cada cédula e incluyéndolo)
    para poder calcular `Recencia_ultimo_producto` en la transformación. El
    periodo de referencia es el mismo que usa `extract_cantidad_productos`
    (Periodo de `inac` - 1 mes). Incluirlo es necesario para poder detectar una
    mejora de tenencia ocurrida justo en ese mes (comparándolo contra el mes
    inmediatamente anterior).

    La query SQL se obtiene desde sql/cantidad_productos.sql (la misma que usa
    `extract_cantidad_productos` para el snapshot puntual).

    Parameters
    ----------
    inac : pd.DataFrame
        DataFrame de población de inactivos. Debe contener las columnas
        'Periodo' (str YYYYMM) e 'ID'.
    con_bi : str
        String de conexión pyodbc a BodegaCorporativa.
    meses_atras : int, optional
        Cantidad de meses de historia a traer antes del periodo de referencia.
        Default 12.
    tamanio_lote : int, optional
        Número máximo de cédulas por lote de consulta. Default 2000.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario {'tenencia_historica': DataFrame} con columnas Periodo, Id,
        Numcantidadproductos, con un registro por cédula y mes histórico.

    Raises
    ------
    FileNotFoundError
        Si el archivo cantidad_productos.sql no existe en sql/.
    ValueError
        Si no se obtuvieron resultados para ningún periodo o lote consultado.
    """
    path_query = Path.cwd() / "sql" / "cantidad_productos.sql"

    if not path_query.exists():
        raise FileNotFoundError(f"La consulta {path_query.name} no existe en sql/")

    base_query = path_query.read_text(encoding="utf-8", errors="coerce")

    df_list = []

    for periodo, grupo in inac.groupby("Periodo"):
        periodo_referencia = datetime.datetime.strptime(f"{periodo}01", "%Y%m%d") - relativedelta(months=1)

        periodos_historicos = [
            (periodo_referencia - relativedelta(months=i)).strftime("%Y-%m-%d")
            for i in range(0, meses_atras)
        ]

        cedulas_str = [str(c) for c in grupo["ID"].unique().tolist()]
        lotes = [
            cedulas_str[i : i + tamanio_lote]
            for i in range(0, len(cedulas_str), tamanio_lote)
        ]

        for periodo_consulta in periodos_historicos:
            query_periodo = base_query.replace("'?'", f"'{periodo_consulta}'")
            conn_bi = None
            try:
                conn_bi = pyodbc.connect(con_bi)
                for lot in lotes:
                    cedulas_consulta = ",".join(f"'{c}'" for c in lot)
                    query_lote = query_periodo.replace("{ids}", cedulas_consulta)
                    df_temp = pd.read_sql(sql=query_lote, con=conn_bi)  # type: ignore
                    df_list.append(df_temp)
                print(f"tenencia_historica -- Periodo: {periodo_consulta} -- Consultado")
            except Exception as e:
                print(f"Error en periodo {periodo_consulta}: {e}")
                raise
            finally:
                if conn_bi is not None:
                    conn_bi.close()

    if not df_list:
        raise ValueError("No se obtuvieron resultados. Verifica la query y los parámetros.")

    return {"tenencia_historica": pd.concat(df_list, axis=0, ignore_index=True)}


def extract_reactivaciones_historicas(
    inac: pd.DataFrame,
    path_reac: str,
) -> dict[str, pd.DataFrame]:
    """
    Lee el Excel maestro de reactivaciones históricas y lo filtra a solo las
    cédulas presentes en `inac`, con Periodo estrictamente anterior al periodo
    de cada cédula. Sirve de insumo para calcular `Distancia_ultima_reactivacion`
    y `Cantidad_Reactivaciones_Previas` en la transformación.

    No hace ninguna consulta SQL: es un archivo Excel en una ruta de red
    (configs/config.yml -> paths_bases -> reactivaciones_historicas).

    Cédulas que no aparezcan en el Excel maestro simplemente no tendrán
    historial de reactivación: la transformación las trata como 0 reactivaciones
    previas y calcula `Distancia_ultima_reactivacion` con el fallback de
    Antiguedad*12, sin necesidad de una rama especial aquí.

    Parameters
    ----------
    inac : pd.DataFrame
        DataFrame de población de inactivos. Debe contener las columnas
        'Periodo' (str YYYYMM) e 'ID'.
    path_reac : str
        Ruta (local o UNC) al Excel maestro de reactivaciones, hoja
        'Base Reactivados 2020-2024'.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario {'reactivaciones_historicas': DataFrame} con columnas
        ['Periodo', 'ID'], solo cédulas de `inac` y periodos previos al de
        cada una.

    Raises
    ------
    FileNotFoundError
        Si `path_reac` no existe.
    """
    path_reac_resuelto = Path(path_reac)

    if not path_reac_resuelto.exists():
        raise FileNotFoundError(f"No se encontró el Excel maestro de reactivaciones en {path_reac}")

    reac_hist = (
        pd.read_excel(path_reac_resuelto, sheet_name="Base Reactivados 2020-2024")
        .rename(columns={"Mes Reactivación": "Fecha", "Cedula": "ID"})
        .assign(
            Periodo=lambda df: df["Fecha"].dt.strftime(date_format="%Y%m").astype(str),
            ID=lambda df: _id_a_str(df["ID"]),
        )
        [["Periodo", "ID"]]
    )

    inac_ids = set(inac["ID"].astype(str).unique().tolist())

    reac_hist = reac_hist[reac_hist["ID"].isin(inac_ids)]

    # Se filtra a periodos estrictamente anteriores al periodo objetivo de cada cédula
    inac_periodos = inac[["ID", "Periodo"]].astype(str).rename(columns={"Periodo": "Periodo_objetivo"})

    reac_hist = (
        reac_hist.merge(right=inac_periodos, how="inner", on="ID")
        .loc[lambda df: df["Periodo"] < df["Periodo_objetivo"], ["Periodo", "ID"]]
        .drop_duplicates(ignore_index=True)
    )

    print(f"reactivaciones_historicas -- {reac_hist['ID'].nunique()} cédulas con historial previo encontrado")

    return {"reactivaciones_historicas": reac_hist}

def extract_fac_rec_gecc(
    inac: pd.DataFrame,
    con_bi: str,
    meses_atras: int = 6,
    tamanio_lote: int = 2000,
) -> dict[str, pd.DataFrame]:
    """
    Extrae el histórico de facturación y recaudo GECC (`meses_atras` meses,
    contados hacia atrás desde el periodo de referencia de cada cédula e
    incluyéndolo) para poder calcular `Factura_Total_Promedio_GECC_6M` y
    `Recaudo_Total_Promedio_GECC_6M` en la transformación (`calcular_promedio_fac_rec`).
    El periodo de referencia es el mismo que usa `extract_cantidad_productos`
    (Periodo de `inac` - 1 mes). Incluirlo es necesario porque la ventana de
    `calcular_promedio_fac_rec` se calcula hacia atrás desde el Periodo
    objetivo de `inac`, así que necesita datos hasta el mes de referencia
    inclusive (mismo criterio que `extract_tenencia_historica`).

    La query SQL se obtiene desde sql/fac_rec_gecc.sql.

    Parameters
    ----------
    inac : pd.DataFrame
        DataFrame de población de inactivos. Debe contener las columnas
        'Periodo' (str YYYYMM) e 'ID'.
    con_bi : str
        String de conexión pyodbc a BodegaCorporativa.
    meses_atras : int, optional
        Cantidad de meses de historia a traer, contados desde el periodo de
        referencia inclusive. Default 6 (debe coincidir con el `meses` que
        use `calcular_promedio_fac_rec` en la transformación).
    tamanio_lote : int, optional
        Número máximo de cédulas por lote de consulta. Default 2000.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario {'fac_rec': DataFrame} con columnas Periodo, Id,
        Vencido_Mes, Cuota_Mes, Valor_Recaudado_Total.

    Raises
    ------
    FileNotFoundError
        Si el archivo fac_rec_gecc.sql no existe en sql/.
    ValueError
        Si no se obtuvieron resultados para ningún periodo o lote consultado.
    """
    path_query = Path.cwd() / "sql" / "fac_rec_gecc.sql"

    if not path_query.exists():
        raise FileNotFoundError(f"La consulta {path_query.name} no existe en sql/")

    base_query = path_query.read_text(encoding="utf-8", errors="coerce")

    df_list = []

    for periodo, grupo in inac.groupby("Periodo"):
        periodo_referencia = datetime.datetime.strptime(f"{periodo}01", "%Y%m%d") - relativedelta(months=1)

        periodos_historicos = [
            (periodo_referencia - relativedelta(months=i)).strftime("%Y-%m-%d")
            for i in range(0, meses_atras)
        ]

        cedulas_str = [str(c) for c in grupo["ID"].unique().tolist()]
        lotes = [
            cedulas_str[i : i + tamanio_lote]
            for i in range(0, len(cedulas_str), tamanio_lote)
        ]

        for periodo_consulta in periodos_historicos:
            # El SQL tiene dos '?' (uno en FAC, uno en REC) — reemplazamos ambos
            query_periodo = base_query.replace("'?'", f"'{periodo_consulta}'")
            conn_bi = None
            try:
                conn_bi = pyodbc.connect(con_bi)
                for lot in lotes:
                    cedulas_consulta = ",".join(f"'{c}'" for c in lot)
                    query_lote = query_periodo.replace("{ids}", cedulas_consulta)
                    df_temp = pd.read_sql(sql=query_lote, con=conn_bi)  # type: ignore
                    df_list.append(df_temp)
                print(f"Fac-Recaudo -- Periodo: {periodo_consulta} -- Consultado")
            except Exception as e:
                print(f"Error en periodo {periodo_consulta}: {e}")
                raise
            finally:
                if conn_bi is not None:
                    conn_bi.close()

    if not df_list:
        raise ValueError("No se obtuvieron resultados. Verifica la query y los parámetros.")

    fac_rec = pd.concat(df_list, axis=0, ignore_index=True).rename(columns={
        "Valor_Vencido": "Vencido_Mes",
        "Valor_Cuota_Mes": "Cuota_Mes",
        "RecaudoTotal": "Valor_Recaudado_Total",
    })

    return {"fac_rec": fac_rec}

def extract_sipas(
    inac:pd.DataFrame, 
    con_bi:str,
    tamanio_lote:int = 2000, 
) -> dict[str,pd.DataFrame]:
    
    path_query = Path.cwd() / "sql" / "sipas.sql"
    
    if not path_query.exists():
        raise FileNotFoundError(f"La consulta {path_query.name} no existe en sql/")
    
    base_query = path_query.read_text(encoding="utf-8", errors="coerce")
    
    # --- Se asigna el periodo de consulta --- # 
    periodo_consulta = (
        datetime.date.today() - relativedelta(months=1)
    ).strftime(format='%Y-%m-%d')
    
    base_query = base_query.replace('?',periodo_consulta)
    
    # --- Se han lotes de consulta --- #
    cedulas_str = [
        str(c) for c in inac['ID'].unique().tolist()
    ]
    
    lotes = [
        cedulas_str[i:i+tamanio_lote] 
        for i in range(0,len(cedulas_str),tamanio_lote)
    ]
    
    # --- Se itera por cada lote de consulta --- #
    resultados=[]
    conn_bi = None
    for lot in lotes:
        try:
            conn_bi = pyodbc.connect(con_bi)
            cedulas_consulta = ','.join(f"'{c}'" for c in lot)
            query_lote = base_query.replace('{ids}',cedulas_consulta)
            df_temp = pd.read_sql(sql=query_lote, con=conn_bi) #type: ignore
            resultados.append(df_temp)
            print(f"Base Sipas -- Periodo: {periodo_consulta} -- Consultado")
        except Exception as e:
            print(f'Error consultado Sipas -- Periodo: {periodo_consulta}')
            raise
        finally:
            if conn_bi is not None:
                conn_bi.close()
    
    if not resultados:
        raise ValueError(f'No se pudo obtener resultado SIPAS --- Periodo: {periodo_consulta}')
    
    # --- Se unen los lotes --- #
    sipas = (
        pd.concat(
            resultados, 
            axis=0, 
            ignore_index=True
        )
        .rename(
            columns={'strIdentificacion':'ID'}
        )
        .astype(
            {
                'ID':int, 
                'Valor_Capitalizado': 'Int64', 
                'Meses_Hasta_Perseverancia': 'Int64', 
            }
        )
    )

    return {'sipas':sipas}

def extract_pqrs(
    inac:pd.DataFrame, 
    con_bi:str, 
    tamanio_lote:int=2000
) -> dict[str,pd.DataFrame]:
    
    path_query = Path.cwd() / "sql" / "pqrs.sql"
    
    if not path_query.exists():
        raise FileNotFoundError(f"La consulta {path_query.name} no existe en sql/")
    
    base_query = path_query.read_text(encoding="utf-8", errors="coerce")
    
    # --- Se asignan los periods de consulta --- # 
    periodo_consulta = (
        datetime.date.today()
    ).strftime(format='%Y-%m-%d')
    
    periodo_consulta_1y = (
        datetime.datetime
        .strptime(periodo_consulta, '%Y-%m-%d') - relativedelta(year=1)
    ).strftime(format='%Y-%m-%d')
    
    # --- Se hacen lotes de consulta --- #
    cedulas_str = [
        str(c) for c in inac['ID'].unique().tolist()
    ]
    
    lotes = [
        cedulas_str[i:i+tamanio_lote] 
        for i in range(0,len(cedulas_str),tamanio_lote)
    ]
    
    # --- Se itera por cada lote de consulta --- #
    resultados=[]
    conn_bi = None
    for lot in lotes:
        try:
            conn_bi = pyodbc.connect(con_bi)
            cedulas_consulta = ','.join(f"'{c}'" for c in lot)
            query_lote = base_query.replace('{ids}',cedulas_consulta)
            df_temp = pd.read_sql(sql=query_lote, 
                                  con=conn_bi, #type: ignore
                                  params= [periodo_consulta_1y,periodo_consulta])
            resultados.append(df_temp)
            print(f"Base pqrs -- Periodo: {periodo_consulta} -- Consultado")
        except Exception as e:
            print(f'Error consultado Base pqrs -- Periodo: {periodo_consulta}')
            raise
        finally:
            if conn_bi is not None:
                conn_bi.close()
    
    if not resultados:
        raise ValueError(f'No se pudo obtener resultado PQRs --- Periodo: {periodo_consulta}')
    
    # --- Se unen los lotes --- #
    pqrs = (
        pd.concat(
            resultados, 
            axis=0, 
            ignore_index=True
        )
        .astype(
            {
                'Identificacion':int, 
                'Cantidad_pqr_ultimo_anno': 'Int64', 
            }
        )
    )

    return {'pqrs':pqrs}

def extract_meses_inactividad(
    inac:pd.DataFrame, 
    con_gcc:str, 
    tamanio_lote:int=2_000
) -> dict[str,pd.DataFrame]:
    
    path_query = Path.cwd() / "sql" / "meses_inactivo.sql"
    
    if not path_query.exists():
        raise FileNotFoundError(f"La consulta {path_query.name} no existe en sql/")
    
    base_query = path_query.read_text(encoding="utf-8", errors="coerce")
    
    # --- Se hacen lotes de consulta --- #
    cedulas_str = [
        str(c) for c in inac['ID'].unique().tolist()
    ]
    
    lotes = [
        cedulas_str[i:i+tamanio_lote] 
        for i in range(0,len(cedulas_str),tamanio_lote)
    ]
    
    # --- Se itera por cada lote de consulta --- #
    resultados=[]
    conn_gcc = None
    for lot in lotes:
        try:
            conn_gcc = pyodbc.connect(con_gcc)
            cedulas_consulta = ','.join(f"'{c}'" for c in lot)
            query_lote = base_query.replace('{ids}',cedulas_consulta)
            df_temp = pd.read_sql(sql=query_lote, 
                                  con=conn_gcc) #type: ignore
            resultados.append(df_temp)
            print("Base meses Inactividad consultada...")
        except Exception as e:
            print('Error consultado meses Inactividad...')
            raise
        finally:
            if conn_gcc is not None:
                conn_gcc.close()
    
    if not resultados:
        raise ValueError(f'No se pudo obtener resultado de meses inactividad...')
    
    # --- Se unen los lotes --- #
    meses_inactividad = (
        pd.concat(
            resultados, 
            axis=0, 
            ignore_index=True
        )
        .astype(
            {
                'CEDULA':int, 
                'Tiempo_Inactividad_Meses': 'Int64', 
            }
        )
    )

    return {'meses_inactividad':meses_inactividad}

def leer_bases(paths_bases: dict[str, str]) -> dict[str, pd.DataFrame]:
    """
    Lee un conjunto de bases insumo desde disco a partir de un diccionario de rutas
    y las retorna en memoria como diccionario {clave: DataFrame}.

    No persiste nada en disco: la persistencia de todas las bases del módulo
    (esta y las que arme cada función de extracción) queda centralizada en
    `export_bases`.

    Parameters
    ----------
    paths_bases : dict[str, str]
        Diccionario {clave: ruta_archivo}. Se lee normalmente desde
        config.yml → paths_bases. Soporta archivos .xlsx, .csv y .parquet.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario {clave: DataFrame} con las bases leídas.

    Raises
    ------
    FileNotFoundError
        Si alguno de los archivos indicados en `paths_bases` no existe.
    ValueError
        Si la extensión de alguno de los archivos no está soportada
        (solo .xlsx, .csv y .parquet).
    """
    lectores = {
        ".xlsx": pd.read_excel,
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
    }

    bases: dict[str, pd.DataFrame] = {}

    print("Leyendo bases insumo...")
    for clave, ruta in paths_bases.items():
        path_archivo = Path(ruta)

        if not path_archivo.exists():
            raise FileNotFoundError(f"No existe el archivo '{path_archivo}' para la base '{clave}'")

        lector = lectores.get(path_archivo.suffix.lower())
        if lector is None:
            raise ValueError(f"Extensión '{path_archivo.suffix}' no soportada para la base '{clave}'")

        bases[clave] = lector(path_archivo)
        print(f"Base '{clave}' -- Leída ({path_archivo})")

    return bases


def export_bases(
    bases: dict[str, pd.DataFrame],
    periodo: str,
    raw_path: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Persiste cada DataFrame del diccionario `bases` como archivo Parquet en la
    capa raw del proyecto, dentro de una carpeta con el periodo de ejecución
    (YYYYMM), ej. data/raw/202608/cantidad_productos.parquet.

    Cada corrida crea (o reutiliza) su propia carpeta `raw_path/periodo/`, así
    que dos corridas del mismo mes se sobreescriben entre sí a propósito (la
    carpeta representa "la última extracción de ese periodo", no un historial
    de corridas).

    Itera clave/valor del diccionario, imprime la confirmación de cada
    exportación, y retorna el mismo diccionario para que el siguiente módulo
    (transformación) lo consuma, ej. `base1 = bases["clave"]`.

    Parameters
    ----------
    bases : dict[str, pd.DataFrame]
        Diccionario {clave: DataFrame} a persistir.
    periodo : str
        Periodo de ejecución en formato YYYYMM (ej. '202608'). Nombra la
        carpeta donde se persisten todas las bases de esta corrida.
    raw_path : str or None, optional
        Ruta base de salida personalizada. Si es None, exporta bajo data/raw/
        relativo a la raíz del proyecto.

    Returns
    -------
    dict[str, pd.DataFrame]
        El mismo diccionario recibido (sin modificar), ya persistido en raw.

    Note:
        Antes de escribir cada parquet, las columnas de tipo ``object`` se
        sanean a ``str`` sobre una copia (preservando los NaN). Esto es
        necesario porque algunos insumos (ej. Excel consolidados mes a mes)
        traen columnas con tipos mezclados (texto y `datetime` en la misma
        columna), que pyarrow no puede inferir y hacen fallar `to_parquet`.
        El diccionario en memoria que se retorna conserva los tipos
        originales; el saneo solo afecta al archivo persistido.
    """
    path_raiz = Path(raw_path) if raw_path else Path(__file__).parents[2] / "data" / "raw"
    path_salida = path_raiz / periodo
    path_salida.mkdir(parents=True, exist_ok=True)

    print(f"Exportando bases - raw/{periodo}")
    for clave, df in bases.items():
        nombre_archivo = f"{clave}.parquet"

        df_export = df.copy()
        for col in df_export.select_dtypes(include="object").columns:
            df_export[col] = df_export[col].map(lambda x: str(x) if pd.notna(x) else x)

        df_export.to_parquet(path_salida / nombre_archivo, engine="pyarrow", index=False)
        print(f"Base '{clave}' -- Correctamente exportada en raw ({path_salida / nombre_archivo})")

    return bases

# ─── Orquestadora ─────────────────────────────────────────────────────────────

def extract_raw(
    con_bi: str,
    con_gcc: str,
    paths_bases: dict[str, str] | None = None,
    tamanio_lote: int = 2000,
    raw_path: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Orquesta el módulo de extracción raw del proyecto.

    Función de entrada principal del módulo. 'poblacion_inactivos' ya no se
    lee de Excel: sale de consultar BodegaCorporativa (`extract_inactivos`),
    que siempre trae un único periodo (el mes en curso). Las demás bases que
    antes venían de Excel (features_inactivos, enriquecimiento_360,
    consolidado_reactivaciones) quedaron deprecadas por completo: sus
    variables ya están cubiertas por consultas SQL existentes
    (meses_inactividad, sipas, pqrs, v_360, y la CTE de intenciones_retiro_1y
    dentro de demografica) y no se leen más — si `paths_bases` todavía las
    trae, se ignoran. Solo 'clv' se sigue leyendo de Excel.

    Con `inac` ya en mano, usa 'poblacion_inactivos' para consultar en
    BodegaCorporativa cantidad de productos (`extract_cantidad_productos`),
    histórico de tenencia (`extract_tenencia_historica`), vista 360
    (`extract_v_360`), demográfica (`extract_demografica`) y
    facturación/recaudo GECC (`extract_fac_rec_gecc`), además de leer el
    Excel maestro de reactivaciones históricas
    (`extract_reactivaciones_historicas`), junta todo en un único diccionario
    y lo persiste en raw (`export_bases`), bajo una carpeta con el periodo de
    ejecución (YYYYMM).

    Parameters
    ----------
    con_bi : str
        String de conexión pyodbc a BodegaCorporativa.
    paths_bases : dict[str, str] or None, optional
        Diccionario {clave: ruta_archivo} de bases insumo a leer. Si es None,
        se toma de config.yml → paths_bases. La clave 'poblacion_inactivos',
        si viene incluida, se ignora (siempre sale de `extract_inactivos`).
    tamanio_lote : int, optional
        Número máximo de cédulas por lote de consulta. Default 2000.
    raw_path : str or None, optional
        Ruta base de salida personalizada para la capa raw. Si es None,
        exporta bajo data/raw/ relativo a la raíz del proyecto.

    Returns
    -------
    dict[str, pd.DataFrame]
        Diccionario {clave: DataFrame} con todas las bases del módulo
        (poblacion_inactivos, clv, cantidad_productos, tenencia_historica,
        v_360, demografica, reactivaciones_historicas, fac_rec, sipas, pqrs,
        meses_inactividad), ya persistidas en raw/{periodo}/. Listo para que
        transformación haga `base1 = bases["poblacion_inactivos"]`.
    """
    print("Iniciando extracción raw...")

    cfg = load_config()
    paths_bases_resueltos: dict[str, str] = dict(
        paths_bases if paths_bases is not None else cfg["paths_bases"]
    )
    path_reac: str = cfg["paths"]["path_reac"]

    # 'poblacion_inactivos' ya no se lee de Excel: sale de extract_inactivos (SQL).
    paths_bases_resueltos.pop("poblacion_inactivos", None)

    bases = leer_bases(paths_bases=paths_bases_resueltos)

    inac = extract_inactivos(con_bi=con_bi)
    inac["ID"] = inac["ID"].astype(str)
    inac["Periodo"] = inac["Periodo"].astype(str)

    periodo_ejecucion = inac["Periodo"].iloc[0]

    # Se conserva la clave/forma histórica ('poblacion_inactivos' con columna
    # 'Cedula') para no romper build_analytic_score todavía — ese ajuste queda
    # para cuando conectemos transformación con este nuevo extract.
    bases["poblacion_inactivos"] = inac.rename(columns={"ID": "Cedula"})

    bases.update(
        extract_cantidad_productos(
            inac=inac,
            con_bi=con_bi,
            tamanio_lote=tamanio_lote,
        )
    )

    bases.update(
        extract_tenencia_historica(
            inac=inac,
            con_bi=con_bi,
            tamanio_lote=tamanio_lote,
        )
    )

    bases.update(
        extract_v_360(
            inac=inac,
            con_bi=con_bi,
            tamanio_lote=tamanio_lote,
        )
    )

    bases.update(
        extract_demografica(
            inac=inac,
            con_bi=con_bi,
            tamanio_lote=tamanio_lote,
        )
    )

    bases.update(
        extract_reactivaciones_historicas(
            inac=inac,
            path_reac=path_reac,
        )
    )
    
    bases.update(
        extract_fac_rec_gecc(
            inac=inac, 
            con_bi=con_bi, 
            tamanio_lote=tamanio_lote, 
        )
    )
    
    bases.update(
        extract_sipas(
            inac=inac, 
            con_bi=con_bi, 
            tamanio_lote=tamanio_lote
        )
    )
    
    bases.update(
        extract_pqrs(
            inac=inac,
            con_bi=con_bi, 
            tamanio_lote=tamanio_lote, 
        )
    )
    
    bases.update(
        extract_meses_inactividad(
            inac=inac, 
            con_gcc=con_gcc, 
            tamanio_lote=tamanio_lote
        )
    )

    bases = export_bases(bases=bases, periodo=periodo_ejecucion, raw_path=raw_path)

    print("Extracción raw -- Finalizada")

    return bases