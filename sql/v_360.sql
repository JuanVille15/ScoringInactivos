SELECT
	Periodo as [Periodo],
	Identificacion as [Identificacion],
	Tipo_Cliente_Bancoomeva as [Tipo_cliente_bancoomeva],
	Tipo_Cliente_Prepagada as [Tipo_cliente_prepagada],
	Tipo_Cliente_Seguros as [Tipo_cliente_seguros],
	Tipo_Cliente_Adicionales as [Tipo_cliente_adicionales],
	ValorPerseverancia as [Valorperseverancia],
	Turnos_En_Oficinas_Total_Ult12Meses as [Turnos_En_Oficinas_Total_Ult12Meses]
FROM Operaciones.dbo.ConsultaIntegral360
WHERE
	Operaciones.$partition.pf_mes(dtmFechaInsercion) = Operaciones.$partition.pf_mes('?')
	AND Identificacion IN ({ids})
