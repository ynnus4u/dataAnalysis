"""
custom timing decorator take from https://github.com/realpython/materials/blob/master/pandas-fast-flexible-intuitive/tutorial/timer.py
"""

import functools
import gc
import itertools
import sys
import pandas as pd
import numpy as np
from timeit import default_timer as _timer


def timeit(_func=None, *, repeat=3, number=1000, file=sys.stdout):
    """Decorator: prints time from best of `repeat` trials.

    Mimics `timeit.repeat()`, but avg. time is printed.
    Returns function result and prints time.

    You can decorate with or without parentheses, as in
    Python's @dataclass class decorator.

    kwargs are passed to `print()`.

    >>> @timeit
    ... def f():
    ...     return "-".join(str(n) for n in range(100))
    ...
    >>> @timeit(number=100000)
    ... def g():
    ...     return "-".join(str(n) for n in range(10))
    ...
    """

    _repeat = functools.partial(itertools.repeat, None)

    def wrap(func):
        @functools.wraps(func)
        def _timeit(*args, **kwargs):
            # Temporarily turn off garbage collection during the timing.
            # Makes independent timings more comparable.
            # If it was originally enabled, switch it back on afterwards.
            gcold = gc.isenabled()
            gc.disable()
 
            try:
                # Outer loop - the number of repeats.
                trials = []
                for _ in _repeat(repeat):
                    # Inner loop - the number of calls within each repeat.
                    total = 0
                    for _ in _repeat(number):
                        start = _timer()
                        result = func(*args, **kwargs)
                        end = _timer()
                        total += end - start
                    trials.append(total)

                # We want the *average time* from the *best* trial.
                # For more on this methodology, see the docs for
                # Python's `timeit` module.
                #
                # "In a typical case, the lowest value gives a lower bound
                # for how fast your machine can run the given code snippet;
                # higher values in the result vector are typically not
                # caused by variability in Python’s speed, but by other
                # processes interfering with your timing accuracy."
                best = min(trials) / number
                print(
                    "Best of {} trials with {} function"
                    " calls per trial:".format(repeat, number)
                )
                print(
                    "Function `{}` ran in average"
                    " of {:0.3f} seconds.".format(func.__name__, best),
                    end="\n\n",
                    file=file,
                )
            finally:
                if gcold:
                    gc.enable()
            # Result is returned *only once*
            return result

        return _timeit

    # Syntax trick from Python @dataclass
    if _func is None:
        return wrap
    else:
        return wrap(_func)


def custom_aggregation(input_df, grouping_key, aggregations, other_columns = [], as_index = False):
    output = input_df.groupby(grouping_key,as_index=as_index).aggregate(aggregations)
    if len(other_columns):
        output[other_columns] = ['Total']*len(other_columns)
    return output


@timeit(repeat=1, number=1)
def solution(input_df, aggregations, current_grouping_key ):
    candidate_grouping_keys = [ ["counter_party"],["legal_entity"],["counter_party", "legal_entity"], ["tier"]]
    cols = current_grouping_key + list(aggregations.keys())
    aggregations_list = [None]*len(candidate_grouping_keys)
    for i,grouping_key in enumerate(candidate_grouping_keys):
        other_columns = [ x for x in current_grouping_key if x not in grouping_key]
        custom_df = custom_aggregation(input_df,grouping_key,aggregations, other_columns )
        #print(f"{custom_df.columns=}")
        #display(custom_df)
        #display(custom_df[cols]) #display_id = ' '.join(grouping_key)
        aggregations_list[i] = custom_df
        #custom_df = custom_df[cols]

    output_df = pd.concat([input_df] + aggregations_list).reset_index(drop=True)
    return output_df


df1 = pd.read_csv( "dataset1.csv" )
df2 = pd.read_csv( "dataset2.csv" )
input_df = pd.merge(df1, df2, how='left', on='counter_party')
#print(input_df)

"""Intermediate Output 1.1"""
current_grouping_key = ["legal_entity", "counter_party", "tier"]
result_1 = input_df.groupby(current_grouping_key).aggregate({'rating': 'max'})
#print(result_1)

"""Intermediate Output 1.2"""
result_2 = pd.pivot_table(input_df, values='value', index=current_grouping_key,  columns=['status'], aggfunc=np.sum).fillna(0)
#print(result_2)

"""Intermediate Output 2"""
result = pd.merge(result_1, result_2, how='left', on=current_grouping_key)
result_post_reset = result.reset_index()

"""FINAL OUTPUT: As concatenate of several dataframes"""
aggregations = {'rating': 'max', 'ACCR': 'sum', 'ARAP': 'sum'}
final_1 = solution(result_post_reset, aggregations,current_grouping_key ); print(final_1)

'''
other attempted solution
@timeit(repeat=1, number=1)
def solution_2(input_dataframe, aggregationss ):
    output_df1 = custom_aggregation(input_dataframe,["counter_party", "legal_entity"], aggregationss,["tier"], True )
    output_df2 = custom_aggregation(input_dataframe,["tier"], aggregationss,["counter_party", "legal_entity"], True )
    
    output_df3 = custom_aggregation(output_df1,["counter_party"], aggregationss,["legal_entity", "tier"], True)
    output_df4 = custom_aggregation(output_df1,["legal_entity"], aggregationss,["counter_party", "tier"], True)
    
    output_df = pd.concat([
            input_dataframe.reset_index(),
            output_df1.reset_index(), 
            output_df2.reset_index(),
            output_df3.reset_index(), 
            output_df4.reset_index() ])
    
    output_df = output_df.reset_index().drop(["index"], axis=1)
    return output_df
#final_2 = solution_2(result, aggregations );print(final_2)
'''
