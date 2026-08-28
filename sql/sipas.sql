SELECT
	strIdentificacion, 
	SUM(Valor_Acum_Capitalizado) AS [Valor_Capitalizado],
	CAST(DATEDIFF(MONTH,  CAST(GETDATE() AS DATE), MAX(dtmFechaPerseverancia)) AS INT) AS [Meses_Hasta_Perseverancia]
FROM Operaciones.Solidaridad.tblPlanesSolidaridadAsociadosInactivos
WHERE
    Operaciones.$partition.pf_mes(dtmFechaInsercion) =
    Operaciones.$partition.pf_mes('?')
	AND strIdentificacion IN ({ids})
GROUP BY
	strIdentificacion;