from vision.marker_diagnostics import marker_diagnostics, measurement_readiness


def test_diagnostics_identify_specific_unstable_marker_without_claiming_accuracy():
    markers=[{'candidate_index':4,'corners_source':'actual_calibration'},
             {'candidate_index':7,'corners_source':'actual_calibration','detection_pass':'irregular_contour_recovery'}]
    measurement={'calibration':{'per_marker_max_error_mm':[2,12],'leave_one_out_dimension_change_mm':[5,80]}}
    rows=marker_diagnostics(markers,measurement)
    assert rows[0]['label']=='C1' and rows[0]['candidate_index']==4
    assert rows[0]['requires_review'] is False
    assert rows[1]['label']=='C2' and len(rows[1]['reasons'])==3
    assert rows[1]['requires_review'] is True
    assert 'accuracy' not in rows[0]


def test_missing_calibration_does_not_invent_zero_errors():
    rows=marker_diagnostics([{}],{})
    assert rows[0]['max_square_fit_error_mm'] is None
    assert rows[0]['leave_one_out_change_mm'] is None
    assert rows[0]['corners_source']=='unverified_candidate'


def test_failed_boundary_keeps_detected_count_but_not_calibration_count():
    result=measurement_readiness({'detected_marker_count':4,'markers_on_target_count':None,
                                 'boundary_refinement':{'quad_original_px':None},'measurement':{'status':'MANUAL_REQUIRED'}})
    assert result['detected_marker_count']==4
    assert result['markers_on_target_count'] is None
    assert result['calibration_marker_count']==0
    assert len(result['next_actions'])==2


def test_markers_outside_target_get_specific_advice():
    result=measurement_readiness({'detected_marker_count':4,'markers_on_target_count':2,
                                 'boundary_refinement':{'quad_original_px':[[0,0],[1,0],[1,1],[0,1]]}})
    assert result['boundary_available']
    assert '同一块构件' in result['next_actions'][0]
    assert result['calibration_marker_count']==0


def test_complete_measurement_has_no_retake_advice():
    result=measurement_readiness({'detected_marker_count':4,'markers_on_target_count':4,
                                 'boundary_refinement':{'quad_original_px':[[0,0],[1,0],[1,1],[0,1]]},
                                 'measurement':{'calibration_corners_px':[[],[],[],[]]}})
    assert result['calibration_marker_count']==4 and result['next_actions']==[]


def test_missing_metadata_is_unknown_not_zero_detection():
    result=measurement_readiness({})
    assert result['detected_marker_count'] is None
