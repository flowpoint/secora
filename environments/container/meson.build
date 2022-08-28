podman = find_program('podman')
#params = files('file1', 'dir/file2')


#configure_file('Dockerfile')
#

custom_target('build podman container',
   build_always : true,
   input : 'Dockerfile',
   output :  'container.tar',
   command : [ podman, 'build', '--label', 'secora_dev', '--tag', 'flowpoint/secora_dev:0.0.1', '--output', 'type=tar,dest=@OUTPUT@', '-f', '@INPUT@' ],
   #command : [ 'echo', '@OUTPUT@'],
   #install : false,
   build_by_default : true)

#r = run_command(podman, 'build', '--label', 'secora_dev', '--tag', 'flowpoint/secora_dev:0.0.1', '-f', 'Dockerfile', check: true)

#output = r.stdout().strip()

# configure_file(input : 'Dockerfile',
#   output : 'Dockerfile',
#   configuration : configuration_data())
# 
# r = run_target('podman', command : ['podman', 'build', '--label', 'secora_dev', '--tag', 'flowpoint/secora_dev:0.0.1', '-f', 'Dockerfile'])
#run_target('podman', command : r)
# custom_target('copy script',
#   input : 'Dockerfile',
#   output :  'Dockerfile',
#   command : ['cp', '@INPUT@', '@OUTPUT@'],
#   install : false,
#   build_by_default : true)
