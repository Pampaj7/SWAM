import matlab.engine

def run_matlab_script():
    # chatgpt is dumb
    eng = matlab.engine.start_matlab()
    eng.run('lr_bc.m', nargout=0)
    eng.quit()


run_matlab_script()