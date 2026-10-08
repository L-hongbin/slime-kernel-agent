import copy
import pytest
from examples.kernel_agent.kernel_coverage import _compute_coverage

def sample(custom=600., total=1000., reference=1., candidate=2.):
    return {'reference_runtime': reference, 'kernel_runtime': candidate,
            'metadata': {'custom_kernel_cuda_time_in_profiling_us': custom,
                         'total_kernel_run_time_in_profiling_us': total}}

@pytest.mark.parametrize('custom,total,candidate', [(600,1000,2),(600,1000,.5),(1000,1000,2),(200,1900,2)])
def test_ratio_and_shared_diagnostics(custom,total,candidate):
    x=sample(custom,total,candidate=candidate); before=copy.deepcopy(x)
    new=_compute_coverage(x,{'coverage_reward_type':'gated_time_coverage'})
    old=_compute_coverage(x,{'coverage_reward_type':'time_coverage'})
    speed=_compute_coverage(x,{'coverage_reward_type':'capped_speed_auxiliary'})
    assert new['coverage']==pytest.approx(custom/total)==old['coverage']
    assert {k:v for k,v in new.items() if k!='coverage'}=={k:v for k,v in speed.items() if k!='coverage'}
    assert x==before

@pytest.mark.parametrize('x', [sample(custom=0), sample(custom=-1), sample(custom=1001),
    sample(custom=True), sample(total=0), sample(total=float('inf')), sample(reference=-1),
    sample(reference=float('nan')), sample(reference=None), sample(candidate=0),
    sample(candidate=float('inf')), sample(candidate=True)])
def test_gate_identical_to_speed(x):
    new=_compute_coverage(x,{'coverage_reward_type':'gated_time_coverage'})
    speed=_compute_coverage(x,{'coverage_reward_type':'capped_speed_auxiliary'})
    assert new['coverage']==speed['coverage']==0
    assert new['coverage_invalid_reason']==speed['coverage_invalid_reason']

def test_measurement_flag():
    x=sample(); x['metadata']['coverage_measurement_valid']=False
    assert _compute_coverage(x,{'coverage_reward_type':'gated_time_coverage'})['coverage']==0

def test_zero_h_still_eligible():
    x=sample(200,1900)
    new=_compute_coverage(x,{'coverage_reward_type':'gated_time_coverage'})
    assert new['coverage_reference_fraction']==0
    assert new['coverage']==pytest.approx(200/1900)
