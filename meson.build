project('secora')

subdir('environments')

#r = run_command(pytest)
##run_target('environments', )
#library('secora')

pytest = find_program('pytest', required : false)

if pytest.found()
  test('pytest', pytest, args : ['-n', 'auto', '/root/secora/'], timeout : 600, is_parallel : false)
endif
