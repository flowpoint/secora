project('secora')

subdir('environments')

#r = run_command(pytest)
##run_target('environments', )
#library('secora')

pytest = find_program('pytest', required : false)

if pytest.found()
  test('pytest_nocuda', pytest, args : ['-n', 'auto', '/root/secora/', '--nocuda'], timeout : 600, is_parallel : false)
  test('pytest_cuda', pytest, args : ['-n', '1', '/root/secora/', '--nosimple'], timeout : 600, is_parallel : false)
endif


