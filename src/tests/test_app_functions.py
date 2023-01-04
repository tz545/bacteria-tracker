import numpy as np 

from dash_app.app_functions import add_cell, remove_cell, forward_prop_cells
from utils import Shape

def test_remove_cell_match():
	mouse_click = {'points':[{'curveNumber': 290, 'pointNumber': 0, 'x':0.1, 'y':0.1}]}

	cells = {0:{'points': [[0,0]]}}

	removed = remove_cell(cells, mouse_click)
	assert removed[0] == {}
	assert removed[1] == 0


def test_remove_cell_no_match():
	mouse_click = {'points':[{'curveNumber': 290, 'pointNumber': 0, 'x':0.9, 'y':0.1}]}

	cells = {0:{'points': [[0,0]]}}
	assert remove_cell(cells, mouse_click) is None


def test_add_cell_adds_square():
	lasso_select = {'points': [], 'lassoPoints': {'x': [0.9, 0.9, 4.1, 4.1], 'y': [0.9, 4.1, 0.9, 4.1]}}
	in_select = {(1, 1), (1, 2), (1, 3), (1, 4),\
				 (2, 1), (2, 2), (2, 3), (2, 4),\
				 (3, 1), (3, 2), (3, 3), (3, 4),\
				 (4, 1), (4, 2), (4, 3), (4, 4)}

	cells = {0:'filler'}
	new_cells, new_no = add_cell(cells, [10, 10], lasso_select)
	assert new_no == 1
	assert set(new_cells[new_no]['points']) == in_select


def test_add_cell_handles_selection_outside_boundary():
	lasso_select = {'points': [], 'lassoPoints': {'x': [0.9, 0.9, 5.1, 5.1], 'y': [0.9, 5.1, 0.9, 5.1]}}
	in_select = {(1, 1), (1, 2), (1, 3), (1, 4),\
				 (2, 1), (2, 2), (2, 3), (2, 4),\
				 (3, 1), (3, 2), (3, 3), (3, 4),\
				 (4, 1), (4, 2), (4, 3), (4, 4)}

	cells = {0:'filler'}
	new_cells, new_no = add_cell(cells, [5, 5], lasso_select)
	new_cells_points = set(new_cells[new_no]['points'])
	assert new_no == 1
	assert new_cells_points == in_select
	assert (0,5) not in new_cells_points
	assert (5,0) not in new_cells_points
	assert (5,5) not in new_cells_points


def test_forward_prop_cells_no_cell2():
	points1 = [[0,0], \
			   [0,1],\
			   [1,0],
			   [1,1],
			   [2,0],
			   [2,1]]

	cell1 = {0:{'points':points1, 'size':len(points1)}}
	cell2 = {}
	new_cell2 = forward_prop_cells(cell1, cell2)
	new_cell2_points = new_cell2[0]['points']

	assert len(new_cell2) == 1
	assert set([tuple(x) for x in new_cell2_points]) == set([tuple(x) for x in points1])


def test_forward_prop_cells_small_cell2():
	points1 = [[0,0], \
			   [0,1],\
			   [1,0],
			   [1,1],
			   [2,0],
			   [2,1]]

	cell1 = {0:{'points':points1, 'size':len(points1)}}
	cell2 = {0:{'points':[[0,0]]}}
	new_cell2 = forward_prop_cells(cell1, cell2)
	new_cell2_points = new_cell2[0]['points']

	assert len(new_cell2) == 1
	assert set([tuple(x) for x in new_cell2_points]) == set([tuple(x) for x in points1])


def test_forward_prop_cells_similar_cell2():
	points1 = [[0,0], \
			   [0,1],\
			   [1,0],
			   [1,1],
			   [2,0],
			   [2,1]]

	points2 = [[0,0], \
			   [0,1],\
			   [1,0],
			   [1,1],
			   [2,0],
			   [2,1],
			   [3,0]]

	cell1 = {0:{'points':points1, 'size':len(points1)}}
	cell2 = {0:{'points':points2}}
	new_cell2 = forward_prop_cells(cell1, cell2)
	new_cell2_points = new_cell2[0]['points']

	assert len(new_cell2) == 1
	assert set([tuple(x) for x in new_cell2_points]) == set([tuple(x) for x in points1])


def test_forward_prop_cells_moved_cell2():

	points1 = [[0,0], \
			   [0,1],\
			   [1,0],
			   [1,1],
			   [2,0],
			   [2,1]]

	points2 = [[0,0], \
			   [0,1],\
			   [1,0],
			   [1,1],
			   [0,2],
			   [1,2]]

	cell1 = {0:{'points':points1, 'size':len(points1)}}
	cell2 = {0:{'points':points2}}
	new_cell2 = forward_prop_cells(cell1, cell2)
	new_cell2_points = new_cell2[0]['points']

	assert len(new_cell2) == 1
	assert set([tuple(x) for x in new_cell2_points]) == set([tuple(x) for x in points2])
