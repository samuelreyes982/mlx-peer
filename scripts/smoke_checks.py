"""Small, explicit graders for functional smoke checks. No model-based judging."""
import json


def load_cases(path):
    value = json.loads(path.read_text())
    cases = value.get('cases')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise ValueError('Expected one to eight smoke cases')
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('Invalid smoke case')
        name, prompt = case.get('id'), case.get('prompt')
        if not isinstance(name, str) or not name or len(name) > 64 or name in seen:
            raise ValueError('Case IDs must be short and unique')
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
            raise ValueError('Invalid smoke prompt')
        if 'same_tokens_as' in case and case['same_tokens_as'] not in seen:
            raise ValueError('Repeat comparison must reference a prior case')
        if 'expected_text' in case and not isinstance(case['expected_text'], str):
            raise ValueError('Expected text must be a string')
        if not any(k in case for k in ('expected_text','expected_json','same_tokens_as','manual_review')):
            raise ValueError('Each case needs an explicit check or manual-review label')
        seen.add(name)
    return cases


def grade(case, generated, previous):
    checks = {'finished_at_eos': generated['finish_reason'] == 'stop',
              'nonempty_answer': bool(generated['response'].strip())}
    if 'expected_text' in case:
        checks['expected_text'] = generated['response'].strip() == case['expected_text']
    if 'expected_json' in case:
        try:
            actual = json.loads(generated['response'])
            # JSON's true and 1 must not compare equal in this check.
            checks['expected_json'] = json.dumps(actual, sort_keys=True) == json.dumps(case['expected_json'], sort_keys=True)
        except (ValueError, TypeError):
            checks['expected_json'] = False
    if 'same_tokens_as' in case:
        prior = next(r for r in previous if r['case_id'] == case['same_tokens_as'])
        checks['same_tokens_after_reset'] = generated['generated_token_ids'] == prior['generated_token_ids']
    return {'checks': checks, 'passed': all(checks.values()),
            'manual_quality_review_required': bool(case.get('manual_review'))}
